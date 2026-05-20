//! Jira provider — issues, sprints, and boards via the Jira REST API.
//!
//! Configuration via environment variables:
//!   - `JIRA_URL`: Base URL (e.g., `https://company.atlassian.net`)
//!   - `JIRA_EMAIL`: User email for Basic Auth
//!   - `JIRA_TOKEN`: API token
//!   - `JIRA_PROJECT`: Default project key (e.g., "PROJ")

use crate::core::providers::{ContextProvider, ProviderItem, ProviderParams, ProviderResult};

const B64_CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

fn simple_base64(input: &[u8]) -> String {
    let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
    for chunk in input.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = chunk.get(1).copied().unwrap_or(0) as u32;
        let b2 = chunk.get(2).copied().unwrap_or(0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(B64_CHARS[((n >> 18) & 63) as usize] as char);
        out.push(B64_CHARS[((n >> 12) & 63) as usize] as char);
        if chunk.len() > 1 {
            out.push(B64_CHARS[((n >> 6) & 63) as usize] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(B64_CHARS[(n & 63) as usize] as char);
        } else {
            out.push('=');
        }
    }
    out
}

pub struct JiraConfig {
    pub base_url: String,
    pub email: String,
    pub token: String,
    pub project: Option<String>,
}

impl JiraConfig {
    pub fn from_env() -> Result<Self, String> {
        let base_url = std::env::var("JIRA_URL").map_err(|_| "JIRA_URL not set")?;
        let email = std::env::var("JIRA_EMAIL").map_err(|_| "JIRA_EMAIL not set")?;
        let token = std::env::var("JIRA_TOKEN").map_err(|_| "JIRA_TOKEN not set")?;
        let project = std::env::var("JIRA_PROJECT").ok();

        Ok(Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            email,
            token,
            project,
        })
    }

    fn auth_header(&self) -> String {
        let credentials = format!("{}:{}", self.email, self.token);
        let encoded = simple_base64(credentials.as_bytes());
        format!("Basic {encoded}")
    }
}

pub struct JiraProvider {
    config: Result<JiraConfig, String>,
}

impl Default for JiraProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl JiraProvider {
    pub fn new() -> Self {
        Self {
            config: JiraConfig::from_env(),
        }
    }
}

impl ContextProvider for JiraProvider {
    fn id(&self) -> &'static str {
        "jira"
    }

    fn display_name(&self) -> &'static str {
        "Jira"
    }

    fn supported_actions(&self) -> &[&str] {
        &["issues", "sprints"]
    }

    fn execute(&self, action: &str, params: &ProviderParams) -> Result<ProviderResult, String> {
        let config = self.config.as_ref().map_err(std::clone::Clone::clone)?;
        match action {
            "issues" => list_issues(config, params),
            "sprints" => list_sprints(config, params),
            _ => Err(format!("Unsupported action: {action}")),
        }
    }

    fn cache_ttl_secs(&self) -> u64 {
        120
    }

    fn requires_auth(&self) -> bool {
        true
    }

    fn is_available(&self) -> bool {
        self.config.is_ok()
    }
}

fn list_issues(config: &JiraConfig, params: &ProviderParams) -> Result<ProviderResult, String> {
    let limit = params.limit.unwrap_or(20);
    let project = params
        .state
        .as_deref()
        .or(config.project.as_deref())
        .unwrap_or("*");

    let jql = if project == "*" {
        "ORDER BY updated DESC".to_string()
    } else {
        format!("project={project} ORDER BY updated DESC")
    };

    let url = format!(
        "{}/rest/api/3/search?jql={}&maxResults={limit}",
        config.base_url,
        urlencoding::encode(&jql)
    );

    let response = ureq::get(&url)
        .header("Authorization", &config.auth_header())
        .header("Accept", "application/json")
        .call()
        .map_err(|e| format!("Jira API error: {e}"))?;

    let text = response
        .into_body()
        .read_to_string()
        .map_err(|e| format!("Jira read error: {e}"))?;
    let body: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| format!("Jira JSON parse error: {e}"))?;

    let total = body["total"].as_u64().unwrap_or(0) as usize;
    let issues = body["issues"].as_array().cloned().unwrap_or_default();

    let items: Vec<ProviderItem> = issues
        .iter()
        .map(|issue| {
            let fields = &issue["fields"];
            ProviderItem {
                id: issue["key"].as_str().unwrap_or_default().to_string(),
                title: fields["summary"].as_str().unwrap_or_default().to_string(),
                state: fields["status"]["name"].as_str().map(String::from),
                author: fields["reporter"]["displayName"].as_str().map(String::from),
                created_at: fields["created"].as_str().map(String::from),
                updated_at: fields["updated"].as_str().map(String::from),
                url: Some(format!(
                    "{}/browse/{}",
                    config.base_url,
                    issue["key"].as_str().unwrap_or_default()
                )),
                labels: fields["labels"]
                    .as_array()
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str().map(String::from))
                            .collect()
                    })
                    .unwrap_or_default(),
                body: fields["description"]
                    .as_str()
                    .map(String::from)
                    .or_else(|| {
                        fields["description"]["content"]
                            .as_array()
                            .map(|_| "[Jira rich text — see web UI]".to_string())
                    }),
            }
        })
        .collect();

    Ok(ProviderResult {
        provider: "jira".into(),
        resource_type: "issues".into(),
        items,
        total_count: Some(total),
        truncated: total > limit,
    })
}

fn list_sprints(config: &JiraConfig, params: &ProviderParams) -> Result<ProviderResult, String> {
    let board_id = params
        .state
        .as_deref()
        .ok_or("Sprint listing requires a board ID via the 'state' parameter")?;

    let limit = params.limit.unwrap_or(5);
    let url = format!(
        "{}/rest/agile/1.0/board/{board_id}/sprint?state=active,future&maxResults={limit}",
        config.base_url
    );

    let response = ureq::get(&url)
        .header("Authorization", &config.auth_header())
        .header("Accept", "application/json")
        .call()
        .map_err(|e| format!("Jira Agile API error: {e}"))?;

    let text = response
        .into_body()
        .read_to_string()
        .map_err(|e| format!("Jira read error: {e}"))?;
    let body: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| format!("Jira JSON parse error: {e}"))?;

    let sprints = body["values"].as_array().cloned().unwrap_or_default();
    let items: Vec<ProviderItem> = sprints
        .iter()
        .map(|s| ProviderItem {
            id: s["id"].as_u64().map_or_else(String::new, |n| n.to_string()),
            title: s["name"].as_str().unwrap_or_default().to_string(),
            state: s["state"].as_str().map(String::from),
            author: None,
            created_at: s["startDate"].as_str().map(String::from),
            updated_at: s["endDate"].as_str().map(String::from),
            url: None,
            labels: vec![],
            body: s["goal"].as_str().map(String::from),
        })
        .collect();

    Ok(ProviderResult {
        provider: "jira".into(),
        resource_type: "sprints".into(),
        items,
        total_count: Some(sprints.len()),
        truncated: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jira_provider_is_unavailable_without_env() {
        let _orig_url = std::env::var("JIRA_URL");
        std::env::remove_var("JIRA_URL");
        std::env::remove_var("JIRA_EMAIL");
        std::env::remove_var("JIRA_TOKEN");

        let provider = JiraProvider::new();
        assert!(!provider.is_available());
        assert_eq!(provider.id(), "jira");
        assert!(provider.requires_auth());
    }

    #[test]
    fn jira_provider_supported_actions() {
        let provider = JiraProvider::new();
        assert!(provider.supported_actions().contains(&"issues"));
        assert!(provider.supported_actions().contains(&"sprints"));
    }
}
