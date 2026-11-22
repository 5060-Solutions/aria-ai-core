//! Live download against the real Hugging Face CDN.
//!
//! Ignored by default: it needs the network and pulls real weights. Run with
//!
//! ```text
//! cargo test --test live_download -- --ignored --nocapture
//! ```
//!
//! This is the test that would have caught a fictional CDN host, a placeholder
//! digest, or a redirect policy that makes every download fail. The unit tests
//! all run against a fake transport and cannot see any of that.

#![cfg(feature = "net")]

use std::sync::Arc;

use aria_ai_core::models::catalog::Catalog;
use aria_ai_core::models::download::{CancelToken, DownloadState, Downloader};
use aria_ai_core::models::http::HttpModelSource;
use aria_ai_core::models::store::{FixedDiskSpace, ModelStore};

fn downloader(dir: &tempfile::TempDir) -> Downloader {
    let store =
        ModelStore::with_probe(dir.path(), Arc::new(FixedDiskSpace(u64::MAX))).expect("store");
    let source = Arc::new(HttpModelSource::new("aria-ai-core/live-test").expect("client"));
    Downloader::new(store, source)
}

/// The smallest real entry in the catalogue: ~2.2 MB, one file, one redirect.
#[test]
#[ignore = "network"]
fn the_smallest_catalogue_model_really_downloads_and_verifies() {
    let catalog = Catalog::bundled().expect("catalog");
    let model = catalog.get("silero-vad-v5").expect("model");

    let dir = tempfile::tempdir().expect("tempdir");
    let dl = downloader(&dir);

    let mut states = Vec::new();
    let path = dl
        .download(model, &CancelToken::new(), &mut |p| {
            if states.last() != Some(&p.state) {
                states.push(p.state.clone());
            }
        })
        .expect("download from the real CDN");

    assert!(path.is_file(), "installed file is missing");
    let size = std::fs::metadata(&path).expect("metadata").len();
    assert_eq!(size, model.size_bytes, "catalogue size is wrong");

    // The digest in catalog.json is the one the CDN actually serves.
    dl.store()
        .verify(model)
        .expect("catalogue digest matches the served bytes");

    assert_eq!(states.last(), Some(&DownloadState::Completed));
    println!("downloaded {size} bytes, digest verified");
}

/// Proves resume works against a server that really honours `Range`, which the
/// fake transport can only simulate.
#[test]
#[ignore = "network"]
fn a_partial_download_resumes_from_the_real_cdn() {
    let catalog = Catalog::bundled().expect("catalog");
    let model = catalog.get("silero-vad-v5").expect("model");

    let dir = tempfile::tempdir().expect("tempdir");
    let dl = downloader(&dir);

    // Stop a little way in, keeping the .part file.
    let cancel = CancelToken::new();
    let err = dl
        .download(model, &cancel, &mut |p| {
            if p.downloaded_bytes > 200_000 {
                cancel.cancel();
            }
        })
        .expect_err("should have been cancelled");
    println!("cancelled as expected: {err}");

    let partial = dl.store().partial_size(model);
    assert!(partial > 0, "a partial file should have been kept");
    println!("resuming from {partial} bytes");

    // Finish it. If the Range request were mishandled the digest would fail.
    dl.download(model, &CancelToken::new(), &mut |_| {})
        .expect("resume");
    dl.store().verify(model).expect("resumed file verifies");
}

/// The multi-file path, against the four real Parakeet members (~660 MB).
#[test]
#[ignore = "network, large"]
fn the_multi_file_parakeet_model_really_downloads_and_verifies() {
    let catalog = Catalog::bundled().expect("catalog");
    let model = catalog.get("parakeet-tdt-0.6b-v2-int8").expect("model");

    let dir = tempfile::tempdir().expect("tempdir");
    let dl = downloader(&dir);

    let installed = dl
        .download(model, &CancelToken::new(), &mut |_| {})
        .expect("download");

    assert!(installed.is_dir(), "parakeet installs as a directory");
    for f in &model.files {
        assert!(installed.join(&f.file_name).is_file(), "{}", f.file_name);
    }
    dl.store().verify(model).expect("every member verifies");
}
