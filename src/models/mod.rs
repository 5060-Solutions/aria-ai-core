//! Model lifecycle: what exists, whether this device can run it, getting it
//! onto disk safely, and getting rid of it again.
//!
//! Everything here is platform-independent. The host supplies the storage
//! root and (optionally) the transport; the crate supplies the catalogue,
//! the gating rules, the resume/verify state machine and the deletion logic.

pub mod capability;
pub mod catalog;
pub mod download;
pub mod store;

#[cfg(feature = "net")]
pub mod http;

pub use capability::{DeviceCapability, DeviceTier, Platform};
pub use catalog::{
    check_host, Catalog, ModelAvailability, ModelDescriptor, ModelKind, ModelRuntime, ALLOWED_HOSTS,
};
pub use download::{
    CancelToken, DownloadProgress, DownloadState, Downloader, ModelSource, SourceStream,
};
pub use store::{
    sha256_bytes, sha256_file, DiskSpaceProbe, FixedDiskSpace, InstalledModel, ModelStore,
};

#[cfg(feature = "disk-space")]
pub use store::SysDiskSpace;

#[cfg(feature = "net")]
pub use http::HttpModelSource;
