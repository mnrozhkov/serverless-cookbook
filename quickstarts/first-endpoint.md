---
title: First Endpoint
category: quickstarts
type: endpoint
runtime: cpu
frameworks:
  - nebius-cli
  - nginx
keywords:
  - endpoint
  - auth
  - token
difficulty: quickstart
---

## Getting started with Serverless AI endpoints: Deploy an nginx server with authentication

Use this quickstart to deploy an authenticated HTTP endpoint and verify request behavior.

Official docs example: [Getting started with Serverless AI endpoints](https://docs.nebius.com/serverless/quickstart/endpoints)

## What this example does

Deploys `nginx:alpine` as a public endpoint with token authentication and tests authorized vs unauthorized requests.

## Why this is useful

It validates endpoint lifecycle, network exposure, and auth behavior before moving to model-serving endpoints.

## Prerequisites

- Nebius CLI is installed and configured (see [Setup](../README.md#setup))
- you are in a tenant group with admin permissions
- quota is available:
  - Compute -> Number of virtual machines
  - Virtual Private Cloud -> Total number of allocations

Check your CLI profile:

```bash
cat ~/.nebius/config.yaml
```

## Runtime / compute

- image: `nginx:alpine`
- platform: `cpu-d3`
- preset: `4vcpu-16gb`
- endpoint startup time: typically around 30 seconds

## Files

- `first-endpoint.md`: this quickstart document

## Quickstart

```bash
export AUTH_TOKEN=$(openssl rand -hex 32)

nebius ai endpoint create \
  --name qs-endpoint-nginx \
  --image nginx:alpine \
  --platform cpu-d3 \
  --preset 4vcpu-16gb \
  --public \
  --container-port 80 \
  --auth token \
  --token "$AUTH_TOKEN"

export ENDPOINT_ID=$(nebius ai endpoint get-by-name \
  --name qs-endpoint-nginx --format jsonpath='{.metadata.id}')
export ENDPOINT_IP=$(nebius ai endpoint get "$ENDPOINT_ID" \
  --format jsonpath='{.status.public_endpoints[0]}')
echo "ENDPOINT_IP=$ENDPOINT_IP"
curl -v "http://$ENDPOINT_IP" -H "Authorization: Bearer $AUTH_TOKEN"
curl -v "http://$ENDPOINT_IP"
```

Note: If your organization uses custom networking, you might need to to specify `--subnet-id`. See [Network and Subnet Selection](../DEVELOPER_GUIDE.md#network-and-subnet-selection) for details.

## Expected output

- request with token succeeds (HTTP 200)
- request without token fails (`401 Unauthorized` or `403 Forbidden`)
- endpoint logs are available with request traces

## How to adapt

- replace `nginx:alpine` with your model-serving image
- update `--container-port` to match your app
- switch auth mode and network visibility for production constraints

## Troubleshooting

- if endpoint remains in `STARTING`, verify image startup and endpoint logs
- if auth appears bypassed, recreate token and retest
- if endpoint IP is missing, re-check endpoint status and quotas

View logs:

```bash
nebius ai endpoint logs "$ENDPOINT_ID" --follow
```

Optional cleanup:

```bash
nebius ai endpoint delete "$ENDPOINT_ID"
```
