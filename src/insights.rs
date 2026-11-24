//! Encrypted on-device storage for call insights (design spec §5).
//!
//! Transcripts and summaries are the most sensitive thing this crate ever
//! produces: the verbatim contents of someone's phone calls. They are stored in
//! `SQLite` with the payload sealed under AES-256-GCM, so a copy of the database
//! file — a device backup, a pulled disk, a misconfigured sync — is not a copy
//! of the conversations.
//!
//! ## What the key is, and is not
//!
//! The 32-byte key comes from the host and is never generated, derived, logged
//! or persisted here. On iOS it belongs in the Keychain with
//! `kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly`; on Android in the
//! Keystore; on desktop in the OS credential store. This crate holds it in
//! memory for the life of the [`InsightStore`] and nowhere else. Lose the key
//! and the rows are unreadable — that is the point, and
//! [`InsightStore::open`] cannot tell a wrong key from a corrupt row until it
//! tries to read one.
//!
//! ## What is left in the clear
//!
//! Deliberately, so the list screen can page and sort without decrypting every
//! row: the call id, the created-at timestamp, the duration, and the status
//! discriminant. Everything with words in it — every transcript segment, every
//! summary, every action item, and the model ids that produced them — is inside
//! the sealed blob.
//!
//! Each row is sealed with its own random 96-bit nonce, and the call id and
//! schema version are bound in as associated data, so a row cannot be lifted
//! from one call and replayed as another's.

use std::path::Path;
use std::sync::Mutex;

use aes_gcm::aead::{Aead, KeyInit, Payload};
use aes_gcm::{Aes256Gcm, Nonce};
use rusqlite::{Connection, OptionalExtension};

use crate::error::{AiError, Result};
use crate::types::{CallInsight, InsightStatus};

/// Bytes of key material required. AES-256.
pub const KEY_LEN: usize = 32;

/// AES-GCM standard nonce length.
const NONCE_LEN: usize = 12;

/// Bumped when the sealed payload's shape changes. Bound in as associated data,
/// so a blob written by an older schema will not silently decrypt under a newer
/// interpretation.
const SCHEMA_VERSION: u32 = 1;

/// A row as the list screen needs it, without decrypting the payload.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "uniffi", derive(uniffi::Record))]
pub struct InsightSummaryRow {
    pub call_id: String,
    pub created_at: i64,
    pub duration_secs: u32,
    /// The status discriminant, e.g. `complete`, `failed`. Not the detail —
    /// a failure reason can quote the conversation, so it lives in the blob.
    pub status: String,
}

/// The encrypted insight database.
pub struct InsightStore {
    conn: Mutex<Connection>,
    cipher: Aes256Gcm,
}

/// Hand-written so the key can never reach a log line through a derived
/// `Debug` on this or on anything holding it.
impl std::fmt::Debug for InsightStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InsightStore")
            .field("cipher", &"<redacted>")
            .finish_non_exhaustive()
    }
}

impl InsightStore {
    /// Open (creating if needed) the database at `path`, sealed under `key`.
    ///
    /// # Errors
    /// [`AiError::Config`] if the key is not [`KEY_LEN`] bytes;
    /// [`AiError::Io`] if the database cannot be opened or migrated.
    pub fn open(path: impl AsRef<Path>, key: &[u8]) -> Result<Self> {
        let path = path.as_ref();
        let cipher = Self::cipher_from(key)?;
        let conn = Connection::open(path)
            .map_err(|e| AiError::io_msg(path, format!("open insight db: {e}")))?;
        Self::init(&conn).map_err(|e| AiError::io_msg(path, format!("migrate: {e}")))?;
        Ok(Self {
            conn: Mutex::new(conn),
            cipher,
        })
    }

    /// An in-memory store, for tests and for a host that wants a scratch
    /// session it never persists.
    ///
    /// # Errors
    /// As [`Self::open`].
    pub fn in_memory(key: &[u8]) -> Result<Self> {
        let cipher = Self::cipher_from(key)?;
        let conn = Connection::open_in_memory()
            .map_err(|e| AiError::Internal(format!("open in-memory insight db: {e}")))?;
        Self::init(&conn).map_err(|e| AiError::Internal(format!("migrate: {e}")))?;
        Ok(Self {
            conn: Mutex::new(conn),
            cipher,
        })
    }

    fn cipher_from(key: &[u8]) -> Result<Aes256Gcm> {
        if key.len() != KEY_LEN {
            return Err(AiError::Config(format!(
                "insight key must be {KEY_LEN} bytes, got {}",
                key.len()
            )));
        }
        Aes256Gcm::new_from_slice(key)
            .map_err(|e| AiError::Config(format!("insight key rejected: {e}")))
    }

    fn init(conn: &Connection) -> rusqlite::Result<()> {
        // FULL rather than NORMAL: an insight is written once, at the end of a
        // call, and losing it to a power cut is worse than the extra fsync.
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "synchronous", "FULL")?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS insights (
                 call_id       TEXT PRIMARY KEY NOT NULL,
                 created_at    INTEGER NOT NULL,
                 duration_secs INTEGER NOT NULL,
                 status        TEXT NOT NULL,
                 schema_version INTEGER NOT NULL,
                 nonce         BLOB NOT NULL,
                 payload       BLOB NOT NULL
             );
             CREATE INDEX IF NOT EXISTS insights_created_at
                 ON insights (created_at DESC);",
        )
    }

    /// Insert or replace one insight.
    ///
    /// # Errors
    /// [`AiError::Internal`] if the payload cannot be sealed or written.
    pub fn put(&self, insight: &CallInsight) -> Result<()> {
        let plaintext = serde_json::to_vec(insight)
            .map_err(|e| AiError::Internal(format!("serialize insight: {e}")))?;

        let mut nonce_bytes = [0_u8; NONCE_LEN];
        rand::fill(&mut nonce_bytes);
        let nonce = Nonce::from(nonce_bytes);

        let aad = Self::aad(&insight.call_id);
        let sealed = self
            .cipher
            .encrypt(
                &nonce,
                Payload {
                    msg: &plaintext,
                    aad: &aad,
                },
            )
            .map_err(|_| AiError::Internal("sealing the insight failed".into()))?;

        let conn = crate::util::lock(&self.conn);
        conn.execute(
            "INSERT INTO insights
                 (call_id, created_at, duration_secs, status, schema_version, nonce, payload)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
             ON CONFLICT(call_id) DO UPDATE SET
                 created_at = excluded.created_at,
                 duration_secs = excluded.duration_secs,
                 status = excluded.status,
                 schema_version = excluded.schema_version,
                 nonce = excluded.nonce,
                 payload = excluded.payload",
            rusqlite::params![
                insight.call_id,
                insight.created_at,
                insight.duration_secs,
                status_label(&insight.status),
                SCHEMA_VERSION,
                nonce_bytes.to_vec(),
                sealed,
            ],
        )
        .map_err(|e| AiError::Internal(format!("write insight: {e}")))?;
        Ok(())
    }

    /// Fetch and decrypt one insight. `Ok(None)` when there is no such call.
    ///
    /// # Errors
    /// [`AiError::Internal`] on a read failure, a wrong key, or a tampered row.
    pub fn get(&self, call_id: &str) -> Result<Option<CallInsight>> {
        let conn = crate::util::lock(&self.conn);
        let row = conn
            .query_row(
                "SELECT schema_version, nonce, payload FROM insights WHERE call_id = ?1",
                rusqlite::params![call_id],
                |r| {
                    Ok((
                        r.get::<_, u32>(0)?,
                        r.get::<_, Vec<u8>>(1)?,
                        r.get::<_, Vec<u8>>(2)?,
                    ))
                },
            )
            .optional()
            .map_err(|e| AiError::Internal(format!("read insight: {e}")))?;
        drop(conn);

        let Some((version, nonce_bytes, sealed)) = row else {
            return Ok(None);
        };
        if version != SCHEMA_VERSION {
            return Err(AiError::Internal(format!(
                "insight {call_id} was written by schema {version}, this build speaks {SCHEMA_VERSION}"
            )));
        }
        if nonce_bytes.len() != NONCE_LEN {
            return Err(AiError::Internal(format!(
                "insight {call_id} has a malformed nonce"
            )));
        }

        let nonce = Nonce::try_from(nonce_bytes.as_slice())
            .map_err(|_| AiError::Internal(format!("insight {call_id} has a malformed nonce")))?;
        let plaintext = self
            .cipher
            .decrypt(
                &nonce,
                Payload {
                    msg: &sealed,
                    aad: &Self::aad(call_id),
                },
            )
            // Deliberately indistinguishable: a wrong key and a tampered row
            // are the same answer, and saying which would be an oracle.
            .map_err(|_| {
                AiError::Internal(format!(
                    "insight {call_id} could not be opened — wrong key or altered data"
                ))
            })?;

        let insight = serde_json::from_slice(&plaintext)
            .map_err(|e| AiError::Internal(format!("decode insight: {e}")))?;
        Ok(Some(insight))
    }

    /// List rows newest-first without decrypting anything.
    ///
    /// # Errors
    /// [`AiError::Internal`] if the query fails.
    pub fn list(&self, limit: u32, offset: u32) -> Result<Vec<InsightSummaryRow>> {
        let conn = crate::util::lock(&self.conn);
        let mut stmt = conn
            .prepare(
                "SELECT call_id, created_at, duration_secs, status
                 FROM insights ORDER BY created_at DESC, call_id LIMIT ?1 OFFSET ?2",
            )
            .map_err(|e| AiError::Internal(format!("prepare list: {e}")))?;
        let rows = stmt
            .query_map(rusqlite::params![limit, offset], |r| {
                Ok(InsightSummaryRow {
                    call_id: r.get(0)?,
                    created_at: r.get(1)?,
                    duration_secs: r.get(2)?,
                    status: r.get(3)?,
                })
            })
            .map_err(|e| AiError::Internal(format!("list insights: {e}")))?;
        rows.collect::<rusqlite::Result<Vec<_>>>()
            .map_err(|e| AiError::Internal(format!("read row: {e}")))
    }

    /// How many insights are stored.
    ///
    /// # Errors
    /// [`AiError::Internal`] if the query fails.
    pub fn count(&self) -> Result<u64> {
        let conn = crate::util::lock(&self.conn);
        // SQLite integers are i64, and rusqlite no longer reads them straight
        // into a u64. COUNT(*) is never negative, so the conversion only fails
        // on a corrupt result.
        let n: i64 = conn
            .query_row("SELECT COUNT(*) FROM insights", [], |r| r.get(0))
            .map_err(|e| AiError::Internal(format!("count insights: {e}")))?;
        u64::try_from(n).map_err(|e| AiError::Internal(format!("count insights: {e}")))
    }

    /// Delete one insight. `false` if there was nothing to delete.
    ///
    /// # Errors
    /// [`AiError::Internal`] if the delete fails.
    pub fn delete(&self, call_id: &str) -> Result<bool> {
        let conn = crate::util::lock(&self.conn);
        let n = conn
            .execute(
                "DELETE FROM insights WHERE call_id = ?1",
                rusqlite::params![call_id],
            )
            .map_err(|e| AiError::Internal(format!("delete insight: {e}")))?;
        Ok(n > 0)
    }

    /// Delete everything older than `cutoff` (unix seconds), for a retention
    /// policy. Returns how many rows went.
    ///
    /// # Errors
    /// [`AiError::Internal`] if the delete fails.
    pub fn delete_older_than(&self, cutoff: i64) -> Result<u64> {
        let conn = crate::util::lock(&self.conn);
        let n = conn
            .execute(
                "DELETE FROM insights WHERE created_at < ?1",
                rusqlite::params![cutoff],
            )
            .map_err(|e| AiError::Internal(format!("prune insights: {e}")))?;
        Ok(n as u64)
    }

    /// Drop every row. For "erase my data" in settings.
    ///
    /// # Errors
    /// [`AiError::Internal`] if the delete fails.
    pub fn clear(&self) -> Result<u64> {
        let conn = crate::util::lock(&self.conn);
        let n = conn
            .execute("DELETE FROM insights", [])
            .map_err(|e| AiError::Internal(format!("clear insights: {e}")))?;
        Ok(n as u64)
    }

    /// Associated data: binds a sealed blob to its call and schema, so a row
    /// swapped between calls fails to open rather than decoding as the wrong
    /// conversation.
    fn aad(call_id: &str) -> Vec<u8> {
        let mut v = Vec::with_capacity(call_id.len() + 5);
        v.extend_from_slice(&SCHEMA_VERSION.to_be_bytes());
        v.push(b':');
        v.extend_from_slice(call_id.as_bytes());
        v
    }
}

/// The status discriminant, without any detail that could quote the call.
const fn status_label(status: &InsightStatus) -> &'static str {
    match status {
        InsightStatus::Pending => "pending",
        InsightStatus::Transcribing => "transcribing",
        InsightStatus::Summarizing => "summarizing",
        InsightStatus::Complete => "complete",
        InsightStatus::TranscriptOnly { .. } => "transcript_only",
        InsightStatus::Failed { .. } => "failed",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Speaker, Summary, Transcript, TranscriptSegment};

    fn key(byte: u8) -> [u8; KEY_LEN] {
        [byte; KEY_LEN]
    }

    fn insight(call_id: &str, created_at: i64) -> CallInsight {
        CallInsight {
            call_id: call_id.to_string(),
            created_at,
            duration_secs: 95,
            transcript: Some(Transcript {
                call_id: call_id.to_string(),
                language: "en".into(),
                duration_ms: 95_000,
                segments: vec![TranscriptSegment {
                    speaker: Speaker::Remote,
                    start_ms: 0,
                    end_ms: 4_000,
                    text: "the alarm code is 4417".into(),
                    confidence: 0.9,
                }],
                narrowband: true,
                lost_audio: false,
            }),
            summary: Some(Summary {
                headline: "Gate code shared".into(),
                abstract_text: "The caller read out the alarm code.".into(),
                key_points: vec!["Caller gave the alarm code".into()],
                action_items: Vec::new(),
                topics: vec!["access".into()],
                degraded: false,
            }),
            stt_model_id: Some("whisper-base-q5_1".into()),
            llm_model_id: Some("qwen3-1.7b-instruct-q4_k_m".into()),
            status: InsightStatus::Complete,
        }
    }

    fn store() -> InsightStore {
        InsightStore::in_memory(&key(7)).expect("store")
    }

    #[test]
    fn a_stored_insight_round_trips_intact() {
        let s = store();
        let original = insight("call-1", 1_700_000_000);
        s.put(&original).expect("put");
        let back = s.get("call-1").expect("get").expect("present");
        assert_eq!(back, original);
    }

    #[test]
    fn an_unknown_call_is_none_not_an_error() {
        let s = store();
        assert!(s.get("nope").expect("get").is_none());
    }

    #[test]
    fn a_key_of_the_wrong_length_is_refused_up_front() {
        assert!(matches!(
            InsightStore::in_memory(&[0_u8; 16]),
            Err(AiError::Config(_))
        ));
        assert!(matches!(
            InsightStore::in_memory(&[]),
            Err(AiError::Config(_))
        ));
    }

    #[test]
    fn the_transcript_text_is_not_readable_in_the_database_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("insights.db");
        {
            let s = InsightStore::open(&path, &key(3)).expect("store");
            s.put(&insight("call-1", 1_700_000_000)).expect("put");
        }
        // Read every file SQLite may have written, WAL included.
        let mut raw = Vec::new();
        for entry in std::fs::read_dir(dir.path()).expect("read dir").flatten() {
            if let Ok(bytes) = std::fs::read(entry.path()) {
                raw.extend_from_slice(&bytes);
            }
        }
        let haystack = String::from_utf8_lossy(&raw);
        assert!(
            !haystack.contains("the alarm code is 4417"),
            "transcript text found in plaintext on disk"
        );
        assert!(
            !haystack.contains("Gate code shared"),
            "summary in plaintext"
        );
        assert!(
            !haystack.contains("whisper-base-q5_1"),
            "model id in plaintext"
        );
        // The indexable columns are, by design, in the clear.
        assert!(haystack.contains("call-1"), "call id should be indexable");
    }

    #[test]
    fn the_wrong_key_cannot_read_a_row() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("insights.db");
        {
            let s = InsightStore::open(&path, &key(1)).expect("store");
            s.put(&insight("call-1", 1)).expect("put");
        }
        let wrong = InsightStore::open(&path, &key(2)).expect("store");
        // Listing still works — that metadata is not encrypted.
        assert_eq!(wrong.list(10, 0).expect("list").len(), 1);
        // The contents do not.
        assert!(wrong.get("call-1").is_err());
    }

    #[test]
    fn a_row_replayed_under_another_call_id_will_not_open() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("insights.db");
        let s = InsightStore::open(&path, &key(5)).expect("store");
        s.put(&insight("call-a", 1)).expect("put a");
        s.put(&insight("call-b", 2)).expect("put b");

        // Move call-a's sealed blob and nonce onto call-b's row.
        {
            let conn = crate::util::lock(&s.conn);
            conn.execute(
                "UPDATE insights SET payload = (SELECT payload FROM insights WHERE call_id='call-a'),
                                     nonce   = (SELECT nonce   FROM insights WHERE call_id='call-a')
                 WHERE call_id='call-b'",
                [],
            )
            .expect("swap");
        }
        // AAD binds the blob to its call id, so this is detected rather than
        // silently returning call-a's transcript as call-b's.
        assert!(s.get("call-b").is_err());
        assert!(s.get("call-a").is_ok());
    }

    #[test]
    fn tampering_with_the_sealed_bytes_is_detected() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("insights.db");
        let s = InsightStore::open(&path, &key(9)).expect("store");
        s.put(&insight("call-1", 1)).expect("put");
        {
            let conn = crate::util::lock(&s.conn);
            let mut blob: Vec<u8> = conn
                .query_row(
                    "SELECT payload FROM insights WHERE call_id='call-1'",
                    [],
                    |r| r.get(0),
                )
                .expect("read");
            blob[0] ^= 0xff;
            conn.execute(
                "UPDATE insights SET payload = ?1 WHERE call_id='call-1'",
                rusqlite::params![blob],
            )
            .expect("tamper");
        }
        assert!(s.get("call-1").is_err());
    }

    #[test]
    fn two_writes_of_the_same_call_use_different_nonces() {
        let s = store();
        let i = insight("call-1", 1);
        s.put(&i).expect("first");
        let first: Vec<u8> = {
            let conn = crate::util::lock(&s.conn);
            conn.query_row("SELECT payload FROM insights", [], |r| r.get(0))
                .expect("read")
        };
        s.put(&i).expect("second");
        let second: Vec<u8> = {
            let conn = crate::util::lock(&s.conn);
            conn.query_row("SELECT payload FROM insights", [], |r| r.get(0))
                .expect("read")
        };
        // Same plaintext, same key: identical ciphertext would mean a reused
        // nonce, which is catastrophic for GCM.
        assert_ne!(first, second);
        assert_eq!(s.count().expect("count"), 1, "upsert, not a second row");
    }

    #[test]
    fn listing_is_newest_first_and_pages() {
        let s = store();
        for (i, id) in ["a", "b", "c"].iter().enumerate() {
            s.put(&insight(id, 1_000 + i as i64)).expect("put");
        }
        let all = s.list(10, 0).expect("list");
        assert_eq!(
            all.iter().map(|r| r.call_id.as_str()).collect::<Vec<_>>(),
            vec!["c", "b", "a"]
        );
        assert_eq!(all[0].status, "complete");
        assert_eq!(all[0].duration_secs, 95);

        let page = s.list(1, 1).expect("page");
        assert_eq!(page.len(), 1);
        assert_eq!(page[0].call_id, "b");
    }

    #[test]
    fn a_failure_reason_never_reaches_the_status_column() {
        let s = store();
        let mut i = insight("call-1", 1);
        i.status = InsightStatus::Failed {
            reason: "caller said the passphrase was swordfish".into(),
        };
        s.put(&i).expect("put");
        let row = &s.list(10, 0).expect("list")[0];
        assert_eq!(row.status, "failed");
        assert!(!row.status.contains("swordfish"));
        // But it survives the round trip inside the sealed payload.
        assert_eq!(
            s.get("call-1").expect("get").expect("present").status,
            i.status
        );
    }

    #[test]
    fn deletion_and_retention_remove_rows() {
        let s = store();
        for (i, id) in ["old", "mid", "new"].iter().enumerate() {
            s.put(&insight(id, 100 * (i as i64 + 1))).expect("put");
        }
        assert_eq!(s.count().expect("count"), 3);

        assert!(s.delete("mid").expect("delete"));
        assert!(!s.delete("mid").expect("delete again"));
        assert_eq!(s.count().expect("count"), 2);

        assert_eq!(s.delete_older_than(200).expect("prune"), 1);
        assert_eq!(s.count().expect("count"), 1);
        assert!(s.get("new").expect("get").is_some());

        assert_eq!(s.clear().expect("clear"), 1);
        assert_eq!(s.count().expect("count"), 0);
    }

    #[test]
    fn a_store_reopened_from_disk_still_reads_its_rows() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("insights.db");
        let original = insight("call-1", 1_700_000_000);
        {
            let s = InsightStore::open(&path, &key(4)).expect("store");
            s.put(&original).expect("put");
        }
        let reopened = InsightStore::open(&path, &key(4)).expect("reopen");
        assert_eq!(
            reopened.get("call-1").expect("get").expect("present"),
            original
        );
    }
}
