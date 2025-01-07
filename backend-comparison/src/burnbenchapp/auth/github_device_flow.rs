// Initially from: https://github.com/jakewilkins/gh-device-flow
use std::collections::HashMap;
use std::{fmt, result::Result, thread, time};

use chrono::offset::Utc;
use chrono::{DateTime, Duration};
use serde::{Deserialize, Serialize};

pub fn credential_error(msg: String) -> DeviceFlowError {
    DeviceFlowError::GitHubError(msg)
}

pub fn send_request(
    device_flow: &mut DeviceFlow,
    url: String,
    body: String,
) -> Option<HashMap<String, serde_json::Value>> {
    let client = reqwest::blocking::Client::new();
    let response_struct = client
        .post(&url)
        .header("Accept", "application/json")
        .body(body)
        .send();

    match response_struct {
        Ok(resp) => match resp.json::<HashMap<String, serde_json::Value>>() {
            Ok(hm) => Some(hm),
            Err(err) => {
                device_flow.state = DeviceFlowState::Failure(err.into());
                None
            }
        },
        Err(err) => {
            device_flow.state = DeviceFlowState::Failure(err.into());
            None
        }
    }
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct Credential {
    pub token: String,
    pub expiry: String,
    pub refresh_token: String,
}

impl Credential {
    fn empty() -> Credential {
        Credential {
            token: String::new(),
            expiry: String::new(),
            refresh_token: String::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub enum DeviceFlowError {
    HttpError(String),
    GitHubError(String),
}

impl fmt::Display for DeviceFlowError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            DeviceFlowError::HttpError(string) => write!(f, "DeviceFlowError: {}", string),
            DeviceFlowError::GitHubError(string) => write!(f, "DeviceFlowError: {}", string),
        }
    }
}

impl std::error::Error for DeviceFlowError {}

impl From<reqwest::Error> for DeviceFlowError {
    fn from(e: reqwest::Error) -> Self {
        DeviceFlowError::HttpError(format!("{:?}", e))
    }
}

#[derive(Debug, Clone)]
pub enum DeviceFlowState {
    Pending,
    Processing(time::Duration),
    Success(Credential),
    Failure(DeviceFlowError),
}

#[derive(Clone)]
pub struct DeviceFlow {
    pub host: String,
    pub client_id: String,
    pub scope: String,
    pub user_code: Option<String>,
    pub device_code: Option<String>,
    pub verification_uri: Option<String>,
    pub state: DeviceFlowState,
}

const FIVE_SECONDS: time::Duration = time::Duration::new(5, 0);

impl DeviceFlow {
    pub fn new(client_id: &str, maybe_host: Option<&str>, scope: Option<&str>) -> Self {
        Self {
            client_id: String::from(client_id),
            scope: match scope {
                Some(string) => String::from(string),
                None => String::new(),
            },
            host: match maybe_host {
                Some(string) => String::from(string),
                None => String::from("github.com"),
            },
            user_code: None,
            device_code: None,
            verification_uri: None,
            state: DeviceFlowState::Pending,
        }
    }

    pub fn start(
        client_id: &str,
        maybe_host: Option<&str>,
        scope: Option<&str>,
    ) -> Result<DeviceFlow, DeviceFlowError> {
        let mut flow = DeviceFlow::new(client_id, maybe_host, scope);

        flow.setup();

        match flow.state {
            DeviceFlowState::Processing(_) => Ok(flow.to_owned()),
            DeviceFlowState::Failure(err) => Err(err),
            _ => Err(credential_error(
                "Something truly unexpected happened".into(),
            )),
        }
    }

    pub fn setup(&mut self) {
        let body = format!("client_id={}&scope={}", &self.client_id, &self.scope);
        let entry_url = format!("https://{}/login/device/code", &self.host);

        if let Some(res) = send_request(self, entry_url, body) {
            if res.contains_key("error") && res.contains_key("error_description") {
                self.state = DeviceFlowState::Failure(credential_error(
                    res["error_description"].as_str().unwrap().into(),
                ))
            } else if res.contains_key("error") {
                self.state = DeviceFlowState::Failure(credential_error(format!(
                    "Error response: {:?}",
                    res["error"].as_str().unwrap()
                )))
            } else {
                self.user_code = Some(String::from(res["user_code"].as_str().unwrap()));
                self.device_code = Some(String::from(res["device_code"].as_str().unwrap()));
                self.verification_uri =
                    Some(String::from(res["verification_uri"].as_str().unwrap()));
                self.state = DeviceFlowState::Processing(FIVE_SECONDS);
            }
        };
    }

    pub fn poll(&mut self, iterations: u32) -> Result<Credential, DeviceFlowError> {
        for count in 0..iterations {
            self.update();

            if let DeviceFlowState::Processing(interval) = self.state {
                if count == iterations {
                    return Err(credential_error("Max poll iterations reached".into()));
                }

                thread::sleep(interval);
            } else {
                break;
            }
        }

        match &self.state {
            DeviceFlowState::Success(cred) => Ok(cred.to_owned()),
            DeviceFlowState::Failure(err) => Err(err.to_owned()),
            _ => Err(credential_error(
                "Unable to fetch credential, sorry :/".into(),
            )),
        }
    }

    pub fn update(&mut self) {
        let poll_url = format!("https://{}/login/oauth/access_token", self.host);
        let poll_payload = format!(
            "client_id={}&device_code={}&grant_type=urn:ietf:params:oauth:grant-type:device_code",
            self.client_id,
            &self.device_code.clone().unwrap()
        );

        if let Some(res) = send_request(self, poll_url, poll_payload) {
            if res.contains_key("error") {
                match res["error"].as_str().unwrap() {
                    "authorization_pending" => {}
                    "slow_down" => {
                        if let DeviceFlowState::Processing(current_interval) = self.state {
                            self.state =
                                DeviceFlowState::Processing(current_interval + FIVE_SECONDS);
                        };
                    }
                    other_reason => {
                        self.state = DeviceFlowState::Failure(credential_error(format!(
                            "Error checking for token: {}",
                            other_reason
                        )));
                    }
                }
            } else {
                let mut this_credential = Credential::empty();
                this_credential.token = res["access_token"].as_str().unwrap().to_string();

                if let Some(expires_in) = res.get("expires_in") {
                    this_credential.expiry = calculate_expiry(expires_in.as_i64().unwrap());
                    this_credential.refresh_token =
                        res["refresh_token"].as_str().unwrap().to_string();
                }

                self.state = DeviceFlowState::Success(this_credential);
            }
        }
    }
}

fn calculate_expiry(expires_in: i64) -> String {
    let expires_in = Duration::seconds(expires_in);
    let mut expiry: DateTime<Utc> = Utc::now();
    expiry += expires_in;
    expiry.to_rfc3339()
}
