# Developer Guide

This guide explains how to design and operate workloads using **Serverless AI Jobs and Endpoints**.

It focuses on:

- workload patterns
- operational guardrails
- platform limitations
- troubleshooting common failures

For runnable examples, see the cookbook `README.md`.

---

## Platform Scope (v1)

Serverless AI v1 is the first step toward making Nebius Cloud the easiest way to run GPU workloads without managing infrastructure.

Current capabilities:

- run containerized workloads via CLI or UI
- automatic VM provisioning and cleanup
- pay-per-second compute billing
- SSH access for debugging jobs
- volume mounts for Object Storage and shared filesystems

---

## Workload Model

Serverless workloads fall into  two categories:

**Jobs:**
run-to-completion workloads producing artifacts

Use Jobs for batch workloads:

- training
- fine-tuning
- batch inference
- scientific simulations

**Endpoints:**
continuously running API services

Use Endpoints for serving APIs:

- development and evaluation model serving

---

## Container Strategy

Jobs support images from public and private registries.

Recommended approach:

- build pre-baked images
- pin image digests
- validate runtime compatibility (CUDA/toolchain)

Avoid installing dependencies at runtime when possible.

Bootstrap scripts are useful for quick demos but increase startup time.

---

## Reliability Guardrails

Define the following for every workload class:

- timeouts

Common failure types:

- image pull failures
- dependency errors
- OOM
- data path issues

---

## Cost Guardrails

Start with the smallest viable preset.

Use:

- timeouts
- cost tracking per run

Validate quotas before large demos.

---

## Cleanup and Cost Hygiene

Compute is automatically cleaned up after job completion.

Persistent resources are not.

These include:

- object storage artifacts
- shared filesystems

---

## Network and Subnet Selection

Use explicit `--subnet-id` when:

- your org uses custom VPC/subnet layouts
- route policy or security controls require a specific subnet
- default subnet is unavailable in your project/region

### Default subnet lookup

```bash
nebius vpc subnet get-by-name --name default-subnet \
  --format jsonpath='{.metadata.id}'
```

### Select a non-default subnet

```bash
# list available subnets first
nebius vpc subnet list

# then provide a specific subnet ID in create command
nebius ai job create ... --subnet-id <subnet-id>
```

### Common subnet/network failures

- **No route / unreachable dependencies**: selected subnet has no required egress path.
- **No IP allocation / capacity errors**: subnet does not have sufficient available addresses/resources.
- **Permission failures**: wrong tenant/group/project permissions for the target subnet.

Troubleshooting checks:

```bash
nebius vpc subnet list
nebius ai job get <job-id>
nebius ai logs <job-id>
```

---

## Troubleshooting

## Job creation appears frozen

Symptom:

`nebius ai job create` takes a long time.

Cause:

cold start provisioning and image pull.

Workaround:

track job status:

```bash
nebius ai job get <JOB_ID>
```

stream logs:

```bash
nebius ai logs <JOB_ID> --follow
```

It if takes too long, check logs of the underlying VM.

---

## Job fails with no logs

Cause:

container never started due to entrypoint issues.

Debug technique:

```bash
<command> || (echo FAILED; sleep 86400)
```

SSH into the container and inspect.

---

## SSH connection fails

Most common issues:

- wrong SSH key passed in CLI
- wrong username (use `nebius` username)
- multiple SSH keys loaded, speficy the key to be used

Correct usage:

```bash
--ssh-key "$(cat ~/.ssh/id_rsa.pub)"
ssh nebius@<PUBLIC_IP>
```

---

## Endpoint stuck in STARTING

Typical causes:

- missing environment variables
- container not serving on expected port
- cold start due to large image

Escalate if:

- startup >30 minutes
- endpoint enters ERROR with no logs
