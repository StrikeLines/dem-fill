# GitHub Authentication Guide for Code Assistants

This document provides step-by-step instructions for connecting future instances of code assistants to the StrikeLines GitHub account for the `dem-fill` repository.

## Account Information

- **GitHub Username**: `StrikeLines`
- **Repository**: `https://github.com/StrikeLines/dem-fill.git`
- **Account has 2FA enabled**: YES (requires Personal Access Token)

## Prerequisites

Since the GitHub account has 2FA (Two-Factor Authentication) enabled, you CANNOT use the regular password. You MUST use a Personal Access Token (PAT).

## Step 1: Generate Personal Access Token (if needed)

If you don't have a current valid token, create one:

1. Go to **GitHub.com** and log in
2. Click profile picture (top right) → **Settings**
3. Scroll down → **Developer settings** (left sidebar)
4. **Personal access tokens** → **Tokens (classic)**
5. **Generate new token** → **Generate new token (classic)**
6. Configure the token:
   - **Note**: "RunPod Code Assistant Access" (or similar descriptive name)
   - **Expiration**: 30-90 days (as preferred)
   - **Scopes** - Check these boxes:
     - ✅ `repo` (Full control of private repositories)
     - ✅ `workflow` (Update GitHub Action workflows)
7. Click **Generate token**
8. **IMPORTANT**: Copy the token immediately - format: `ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`

## Step 2: Configure Git Credentials

Run these commands to set up git with the correct user information:

```bash
git config user.name "StrikeLines"
git config user.email "strikelines@users.noreply.github.com"
```

## Step 3: Repository Operations with Authentication

### Clone Repository
```bash
git clone https://StrikeLines:[YOUR_TOKEN]@github.com/StrikeLines/dem-fill.git
```

### Add Remote (if repo already exists)
```bash
git remote add origin https://github.com/StrikeLines/dem-fill.git
```

### Push with Authentication
```bash
git push https://StrikeLines:[YOUR_TOKEN]@github.com/StrikeLines/dem-fill.git [BRANCH_NAME]
```

### Fetch with Authentication
```bash
git fetch https://StrikeLines:[YOUR_TOKEN]@github.com/StrikeLines/dem-fill.git
```

### Pull with Authentication
```bash
git pull https://StrikeLines:[YOUR_TOKEN]@github.com/StrikeLines/dem-fill.git [BRANCH_NAME]
```

## Step 4: Common Workflow Commands

### Initialize New Repository
```bash
# Configure git
git config user.name "StrikeLines"
git config user.email "strikelines@users.noreply.github.com"

# Initialize and add remote
git init
git remote add origin https://github.com/StrikeLines/dem-fill.git

# First commit and push
git add .
git commit -m "Initial commit message"
git push -u https://StrikeLines:[YOUR_TOKEN]@github.com/StrikeLines/dem-fill.git main
```

### Create and Push New Branch
```bash
# Create and switch to new branch
git checkout -b [NEW_BRANCH_NAME]

# Make changes, then commit
git add .
git commit -m "Your commit message"

# Push new branch to GitHub
git push -u https://StrikeLines:[YOUR_TOKEN]@github.com/StrikeLines/dem-fill.git [NEW_BRANCH_NAME]
```

### Force Push to Replace Branch
```bash
git push https://StrikeLines:[YOUR_TOKEN]@github.com/StrikeLines/dem-fill.git [LOCAL_BRANCH]:[REMOTE_BRANCH] --force
```

## Important Notes

### Security
- **NEVER commit this file with a real token** - tokens should be treated as passwords
- Replace `[YOUR_TOKEN]` with the actual token when using commands
- Tokens expire - check expiration date and renew as needed

### Repository Structure
- Main branches: `main`, development branches as created
- **Ignored directories** (already in `.gitignore`):
  - `experiments/` - experiment data
  - `pretrained/` - model files  
  - `temp/` - temporary files
  - `test/` - test data

### Troubleshooting
- If you get "401 Unauthorized" errors, check:
  1. Token is valid and not expired
  2. Token has correct scopes (`repo` and `workflow`)
  3. Username is exactly `StrikeLines`
  4. URL format is correct

### Alternative: Using Git Credential Store
For longer sessions, you can store credentials temporarily:
```bash
git config credential.helper store
# Then on first push/pull, enter username and token
# Subsequent operations won't require re-entering credentials
```

## Example Session

```bash
# Configure git
git config user.name "StrikeLines"
git config user.email "strikelines@users.noreply.github.com"

# Add files and commit
git add .
git commit -m "Update sampling mechanism"

# Push to current branch
git push https://StrikeLines:ghp_your_token_here@github.com/StrikeLines/dem-fill.git
```

## Quick Reference

- **Username**: `StrikeLines`
- **Email**: `strikelines@users.noreply.github.com`  
- **Repo URL**: `https://github.com/StrikeLines/dem-fill.git`
- **Auth Format**: `https://StrikeLines:[TOKEN]@github.com/StrikeLines/dem-fill.git`