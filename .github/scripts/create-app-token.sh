#!/usr/bin/env bash
set -euo pipefail

# Mint a GitHub App installation token without relying on a JavaScript action.

app_id=""
private_key="${PRIVATE_KEY:-}"
repository=""
api_url="${GITHUB_API_URL:-https://api.github.com}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --app-id)
      app_id="$2"
      shift 2
      ;;
    --repository)
      repository="$2"
      shift 2
      ;;
    --api-url)
      api_url="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$app_id" || -z "$repository" ]]; then
  echo "Usage: $0 --app-id <id> --repository <owner/repo> [--api-url <url>]" >&2
  exit 1
fi

if [[ -z "$private_key" ]]; then
  echo "PRIVATE_KEY environment variable must be set" >&2
  exit 1
fi

if [[ "$repository" != */* ]]; then
  echo "Repository must be in <owner/repo> form: $repository" >&2
  exit 1
fi

repo_name="${repository#*/}"

b64url() {
  openssl base64 -A | tr '+/' '-_' | tr -d '='
}

now="$(date +%s)"
iat="$((now - 60))"
exp="$((now + 540))"
header='{"alg":"RS256","typ":"JWT"}'
payload="$(printf '{\"iat\":%s,\"exp\":%s,\"iss\":\"%s\"}' "$iat" "$exp" "$app_id")"

key_file="$(mktemp)"
cleanup() {
  rm -f "$key_file"
}
trap cleanup EXIT
chmod 600 "$key_file"
printf '%s' "$private_key" >"$key_file"

unsigned_token="$(printf '%s' "$header" | b64url).$(printf '%s' "$payload" | b64url)"
signature="$(printf '%s' "$unsigned_token" | openssl dgst -binary -sha256 -sign "$key_file" | b64url)"
jwt="$unsigned_token.$signature"
echo "::add-mask::$jwt"

api_headers=(
  -H "Authorization: Bearer $jwt"
  -H "Accept: application/vnd.github+json"
  -H "X-GitHub-Api-Version: 2022-11-28"
)

installation_json="$(curl -fsSL "${api_headers[@]}" "$api_url/repos/$repository/installation")"
installation_id="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])' <<<"$installation_json")"

request_body="$(python3 -c 'import json,sys; print(json.dumps({"repositories": [sys.argv[1]]}))' "$repo_name")"
token_json="$(curl -fsSL -X POST "${api_headers[@]}" "$api_url/app/installations/$installation_id/access_tokens" -d "$request_body")"
token="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["token"])' <<<"$token_json")"

if [[ -z "$token" || "$token" == "null" ]]; then
  echo "Failed to create a GitHub App installation token for $repository" >&2
  exit 1
fi

echo "::add-mask::$token"

if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
  {
    echo "token=$token"
    echo "installation-id=$installation_id"
  } >>"$GITHUB_OUTPUT"
else
  printf '%s\n' "$token"
fi
