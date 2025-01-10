use arboard::Clipboard;
use burn::serde::{Deserialize, Serialize};
use reqwest;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::{
    fs::{self, File},
    path::{Path, PathBuf},
    thread, time,
};

pub(crate) static CLIENT_ID: &str = "Iv1.692f6a61b6086810";
const FIVE_SECONDS: time::Duration = time::Duration::new(5, 0);
static GITHUB_API_VERSION_HEADER: &str = "X-GitHub-Api-Version";
static GITHUB_API_VERSION: &str = "2022-11-28";

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct Tokens {
    /// Token returned once the Burnbench Github app has been authorized by the user.
    /// This token is used to authenticate the user to the Burn benchmark server.
    /// This token is a short lived token (about 8 hours).
    pub access_token: String,
    /// Along with the access token, a refresh token is provided once the Burnbench
    /// GitHub app has been authorized by the user.
    /// This token can be presented to the Burn benchmark server in order to re-issue
    /// a new access token for the user.
    /// This token is longer lived (around 6 months).
    pub refresh_token: String,
}

#[derive(Debug, Deserialize)]
pub(crate) struct UserInfo {
    pub nickname: String,
}

/// Retrieve cached tokens and refresh them if necessary then save the new tokens.
/// If there is no cached token or if the access token cannot be resfresh then
/// ask for the user to reauthorize the Burnbench github application.
pub(crate) fn get_tokens() -> Option<Tokens> {
    get_tokens_from_cache().map_or_else(
        // no token saved yet
        auth,
        // cached tokens found
        |tokens| {
            if verify_tokens(&tokens) {
                Some(tokens)
            } else {
                refresh_tokens(&tokens).map_or_else(
                    || {
                        println!("⚠ Cannot refresh the access token. You need to reauthorize the Burnbench application.");
                        auth()
                    },
                    |new_tokens| {
                        save_tokens(&new_tokens);
                        Some(new_tokens)
                    })
            }
        },
    )
}

/// Returns the authenticated user name from access token
pub(crate) fn get_username(access_token: &str) -> Option<UserInfo> {
    let client = reqwest::blocking::Client::new();
    let response = client
        .get(format!("{}users/me", USER_BENCHMARK_SERVER_URL))
        .header(reqwest::header::USER_AGENT, "burnbench")
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .header(
            reqwest::header::AUTHORIZATION,
            format!("Bearer {}", access_token),
        )
        .send()
        .ok()?;
    response.json::<UserInfo>().ok()
}

fn auth() -> Option<Tokens> {
    let mut flow = match DeviceFlow::start(CLIENT_ID, None, None) {
        Ok(flow) => flow,
        Err(e) => {
            eprintln!("Error authenticating: {}", e);
            return None;
        }
    };
    println!("🌐 Please visit for following URL in your browser (CTRL+click if your terminal supports it):");
    println!("\n    {}\n", flow.verification_uri.clone().unwrap());
    let user_code = flow.user_code.clone().unwrap();
    println!("👉 And enter code: {}", &user_code);
    if let Ok(mut clipboard) = Clipboard::new() {
        if clipboard.set_text(user_code).is_ok() {
            println!("📋 Code has been successfully copied to clipboard.")
        };
    };
    // Wait for the minimum allowed interval to poll for authentication update
    // see: https://docs.github.com/en/apps/oauth-apps/building-oauth-apps/authorizing-oauth-apps#step-3-app-polls-github-to-check-if-the-user-authorized-the-device
    thread::sleep(FIVE_SECONDS);
    match flow.poll(20) {
        Ok(creds) => {
            let tokens = Tokens {
                access_token: creds.token.clone(),
                refresh_token: creds.refresh_token.clone(),
            };
            save_tokens(&tokens);
            Some(tokens)
        }
        Err(e) => {
            eprint!("Authentication error: {}", e);
            None
        }
    }
}

/// Return the token saved in the cache file
#[inline]
fn get_tokens_from_cache() -> Option<Tokens> {
    let path = get_auth_cache_file_path();
    let file = File::open(path).ok()?;
    let tokens: Tokens = serde_json::from_reader(file).ok()?;
    Some(tokens)
}

/// Returns true if the token is still valid
fn verify_tokens(tokens: &Tokens) -> bool {
    let client = reqwest::blocking::Client::new();
    let response = client
        .get("https://api.github.com/user")
        .header(reqwest::header::USER_AGENT, "burnbench")
        .header(reqwest::header::ACCEPT, "application/vnd.github+json")
        .header(
            reqwest::header::AUTHORIZATION,
            format!("Bearer {}", tokens.access_token),
        )
        .header(GITHUB_API_VERSION_HEADER, GITHUB_API_VERSION)
        .send();
    response.is_ok_and(|resp| resp.status().is_success())
}

fn refresh_tokens(tokens: &Tokens) -> Option<Tokens> {
    println!("Access token must be refreshed.");
    println!("Refreshing token...");
    let client = reqwest::blocking::Client::new();
    let response = client
        .post(format!("{}auth/refresh-token", USER_BENCHMARK_SERVER_URL))
        .header(reqwest::header::USER_AGENT, "burnbench")
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .header(
            reqwest::header::AUTHORIZATION,
            format!("Bearer-Refresh {}", tokens.refresh_token),
        )
        // it is important to explicitly add an empty body otherwise
        // reqwest won't send the request in release build
        .body(reqwest::blocking::Body::from(""))
        .send();
    response.ok()?.json::<Tokens>().ok().inspect(|_new_tokens| {
        println!("✅ Token refreshed!");
    })
}

/// Return the file path for the auth cache on disk
fn get_auth_cache_file_path() -> PathBuf {
    let home_dir = dirs::home_dir().expect("an home directory should exist");
    let path_dir = home_dir.join(".cache").join("burn").join("burnbench");
    #[cfg(test)]
    let path_dir = path_dir.join("test");
    let path = Path::new(&path_dir);
    path.join("token.txt")
}

/// Save token in Burn cache directory and adjust file permissions
fn save_tokens(tokens: &Tokens) {
    let path = get_auth_cache_file_path();
    fs::create_dir_all(path.parent().expect("path should have a parent directory"))
        .expect("directory should be created");
    let file = File::create(&path).expect("file should be created");
    serde_json::to_writer_pretty(file, &tokens).expect("Tokens should be saved to cache file.");
    // On unix systems we lower the permissions on the cache file to be readable
    // just by the current user
    #[cfg(unix)]
    fs::set_permissions(&path, fs::Permissions::from_mode(0o600))
        .expect("permissions should be set to 600");
    println!("✅ Token saved at location: {}", path.to_str().unwrap());
}

#[cfg(test)]
use serial_test::serial;

use crate::burnbenchapp::{auth::github_device_flow::DeviceFlow, USER_BENCHMARK_SERVER_URL};

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::*;
    use std::fs;

    #[fixture]
    fn tokens() -> Tokens {
        Tokens {
            access_token: "unique_test_token".to_string(),
            refresh_token: "unique_refresh_token".to_string(),
        }
    }

    fn cleanup_test_environment() {
        let path = get_auth_cache_file_path();
        if path.exists() {
            fs::remove_file(&path).expect("should be able to delete the token file");
        }
        let parent_dir = path
            .parent()
            .expect("token file should have a parent directory");
        if parent_dir.exists() {
            fs::remove_dir_all(parent_dir).expect("should be able to delete the cache directory");
        }
    }

    #[rstest]
    #[serial]
    fn test_save_token_when_file_does_not_exist(tokens: Tokens) {
        cleanup_test_environment();
        // Ensure the file does not exist
        let path = get_auth_cache_file_path();
        if path.exists() {
            fs::remove_file(&path).unwrap();
        }
        save_tokens(&tokens);
        let retrieved_tokens = get_tokens_from_cache().unwrap();
        assert_eq!(retrieved_tokens.access_token, tokens.access_token);
        assert_eq!(retrieved_tokens.refresh_token, tokens.refresh_token);
        cleanup_test_environment();
    }

    #[rstest]
    #[serial]
    fn test_overwrite_saved_token_when_file_already_exists(tokens: Tokens) {
        cleanup_test_environment();
        save_tokens(&tokens);
        let new_tokens = Tokens {
            access_token: "new_test_token".to_string(),
            refresh_token: "new_refresh_token".to_string(),
        };
        save_tokens(&new_tokens);
        let retrieved_tokens = get_tokens_from_cache().unwrap();
        assert_eq!(retrieved_tokens.access_token, new_tokens.access_token);
        assert_eq!(retrieved_tokens.refresh_token, new_tokens.refresh_token);
        cleanup_test_environment();
    }

    #[test]
    #[serial]
    fn test_return_none_when_cache_file_does_not_exist() {
        cleanup_test_environment();
        let path = get_auth_cache_file_path();
        // Ensure the file does not exist
        if path.exists() {
            fs::remove_file(&path).unwrap();
        }
        assert!(get_tokens_from_cache().is_none());
        cleanup_test_environment();
    }

    #[test]
    #[serial]
    fn test_return_none_when_cache_file_exists_but_is_empty() {
        cleanup_test_environment();
        // Create an empty file
        let path = get_auth_cache_file_path();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("directory tree should be created");
        }
        File::create(&path).expect("empty file should be created");
        assert!(
            get_tokens_from_cache().is_none(),
            "Expected None for empty cache file, got Some"
        );
        cleanup_test_environment();
    }
}
