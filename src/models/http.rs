//! The one and only network capability in this crate (design spec §5.4).
//!
//! HTTPS, rustls, compile-time host allowlist. Nothing else in `aria-ai-core`
//! constructs an HTTP client, which is what makes the "nothing leaves this
//! device" claim auditable rather than aspirational.
//!
//! Redirects are followed manually, never by the HTTP library, and every hop
//! is re-checked against the allowlist before it is dialled. Hugging Face
//! answers a weights request with a 302 to a signed URL on its content CDN, so
//! refusing redirects outright would mean no model could ever be fetched; the
//! property that actually matters — that this crate only ever connects to an
//! allowlisted host — is preserved hop by hop.

use std::time::Duration;

use crate::error::{AiError, Result};
use crate::models::catalog::check_host;
use crate::models::download::{ModelSource, SourceStream};

/// Connect timeout. Model hosts are CDNs; if the handshake is slow, the
/// download is not going to work anyway.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(20);

/// Redirect hops allowed before giving up. HF uses exactly one; the margin is
/// for a CDN that adds another, not for an open-ended chain.
const MAX_REDIRECTS: usize = 4;

/// Blocking HTTPS transport for [`crate::models::download::Downloader`].
#[derive(Debug)]
pub struct HttpModelSource {
    client: reqwest::blocking::Client,
    user_agent: String,
}

impl HttpModelSource {
    /// Build a client.
    ///
    /// # Errors
    /// [`AiError::Config`] if the TLS stack cannot be initialised.
    pub fn new(user_agent: impl Into<String>) -> Result<Self> {
        let user_agent = user_agent.into();
        let client = reqwest::blocking::Client::builder()
            .https_only(true)
            // Followed manually in `open` so every hop is allowlist-checked;
            // the library must never follow one on our behalf.
            .redirect(reqwest::redirect::Policy::none())
            .connect_timeout(CONNECT_TIMEOUT)
            .user_agent(user_agent.clone())
            .build()
            .map_err(|e| AiError::Config(format!("http client: {e}")))?;
        Ok(Self { client, user_agent })
    }

    /// The `User-Agent` this source sends.
    #[must_use]
    pub fn user_agent(&self) -> &str {
        &self.user_agent
    }
}

impl ModelSource for HttpModelSource {
    fn open(&self, url: &str, offset: u64) -> Result<SourceStream> {
        let mut current = url.to_string();
        let mut resp = None;

        for _ in 0..=MAX_REDIRECTS {
            // Checked before every dial, including each redirect target.
            check_host(&current)?;

            let mut req = self.client.get(&current);
            if offset > 0 {
                req = req.header(reqwest::header::RANGE, format!("bytes={offset}-"));
            }
            let r = req.send().map_err(|e| AiError::Download(e.to_string()))?;

            if r.status().is_redirection() {
                let location = r
                    .headers()
                    .get(reqwest::header::LOCATION)
                    .and_then(|v| v.to_str().ok())
                    .ok_or_else(|| {
                        AiError::Download(format!("HTTP {} without a Location", r.status()))
                    })?;
                current = resolve_redirect(&current, location)?;
                continue;
            }

            resp = Some(r);
            break;
        }

        let resp =
            resp.ok_or_else(|| AiError::Download(format!("more than {MAX_REDIRECTS} redirects")))?;

        let status = resp.status();
        if !status.is_success() {
            return Err(AiError::Download(format!("HTTP {status}")));
        }

        // 206 means the server honoured our Range; 200 means it did not and is
        // sending the whole thing from byte 0. Report which so the downloader
        // can reset its hash instead of splicing two ranges together.
        let start_offset = if status == reqwest::StatusCode::PARTIAL_CONTENT {
            offset
        } else {
            0
        };

        let total_bytes = resp.content_length();
        Ok(SourceStream {
            total_bytes,
            start_offset,
            reader: Box::new(resp),
        })
    }
}

/// Turn a `Location` header into the absolute URL to dial next.
///
/// Hugging Face answers with an absolute URL for LFS objects but a root-relative
/// one (`/api/resolve-cache/…`) for small plain files, so handling only absolute
/// targets means small members of a multi-file model can never be fetched. A
/// relative target keeps the current origin, and the caller re-checks the
/// result against the allowlist either way.
///
/// # Errors
/// [`AiError::HostNotAllowed`] if the target cannot be resolved to an absolute
/// HTTPS URL.
fn resolve_redirect(current: &str, location: &str) -> Result<String> {
    if location.starts_with("https://") {
        return Ok(location.to_string());
    }
    // Anything else absolute (http://, file://, …) is refused outright rather
    // than downgraded.
    if location.contains("://") {
        return Err(AiError::HostNotAllowed(format!(
            "redirect to a non-HTTPS scheme: {location}"
        )));
    }
    let rest = current.strip_prefix("https://").ok_or_else(|| {
        AiError::HostNotAllowed(format!("cannot resolve {location} against {current}"))
    })?;
    let authority = rest.split(['/', '?', '#']).next().unwrap_or("");
    if authority.is_empty() {
        return Err(AiError::HostNotAllowed(format!(
            "cannot resolve {location} against {current}"
        )));
    }
    if let Some(path) = location.strip_prefix('/') {
        return Ok(format!("https://{authority}/{path}"));
    }
    // A path-relative target would need the current directory resolved; no
    // real host sends one here, so refuse rather than guess.
    Err(AiError::HostNotAllowed(format!(
        "unsupported relative redirect target: {location}"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_root_relative_redirect_keeps_the_current_origin() {
        // HF really does answer this way for small non-LFS files.
        let next = resolve_redirect(
            "https://huggingface.co/owner/repo/resolve/main/tokens.txt",
            "/api/resolve-cache/models/owner/repo/abc/tokens.txt?etag=%22x%22",
        )
        .expect("resolve");
        assert_eq!(
            next,
            "https://huggingface.co/api/resolve-cache/models/owner/repo/abc/tokens.txt?etag=%22x%22"
        );
        check_host(&next).expect("still allowlisted");
    }

    #[test]
    fn an_absolute_redirect_is_taken_as_given() {
        let next = resolve_redirect(
            "https://huggingface.co/o/r/resolve/main/f.bin",
            "https://us.aws.cdn.hf.co/xet-bridge-us/abc?sig=1",
        )
        .expect("resolve");
        assert_eq!(next, "https://us.aws.cdn.hf.co/xet-bridge-us/abc?sig=1");
    }

    #[test]
    fn a_redirect_cannot_downgrade_the_scheme_or_leave_the_allowlist() {
        assert!(matches!(
            resolve_redirect("https://huggingface.co/a", "http://huggingface.co/a"),
            Err(AiError::HostNotAllowed(_))
        ));
        assert!(matches!(
            resolve_redirect("https://huggingface.co/a", "file:///etc/passwd"),
            Err(AiError::HostNotAllowed(_))
        ));
        // Absolute but off-list: resolved here, refused by the caller's check.
        let next = resolve_redirect("https://huggingface.co/a", "https://evil.example/x")
            .expect("resolved");
        assert!(check_host(&next).is_err(), "must not be dialled");
    }

    #[test]
    fn the_client_builds_and_carries_a_user_agent() {
        let s = HttpModelSource::new("aria-ai-core/test").expect("client");
        assert_eq!(s.user_agent(), "aria-ai-core/test");
    }

    #[test]
    fn off_allowlist_urls_are_refused_without_touching_the_network() {
        let s = HttpModelSource::new("aria-ai-core/test").expect("client");
        assert!(matches!(
            s.open("https://cdn.evil.example/m.bin", 0),
            Err(AiError::HostNotAllowed(_))
        ));
        assert!(matches!(
            s.open("http://huggingface.co/m.bin", 0),
            Err(AiError::HostNotAllowed(_))
        ));
        assert!(matches!(
            s.open("file:///etc/passwd", 0),
            Err(AiError::HostNotAllowed(_))
        ));
    }
}
