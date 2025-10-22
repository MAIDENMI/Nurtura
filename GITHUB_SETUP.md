# GitHub Setup Instructions

Your project is ready to push to GitHub! Follow these steps:

## Step 1: Create a Private GitHub Repository

1. Go to: https://github.com/new
2. Repository settings:
   - **Repository name**: `infant-breathing-monitor` (or your choice)
   - **Description**: "AI-powered infant breathing monitoring using computer vision"
   - **Visibility**: ✅ **Private** (keep it private)
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
3. Click **"Create repository"**

## Step 2: Push Your Code

After creating the repo, GitHub will show you commands. Use these:

```bash
# Add the remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/infant-breathing-monitor.git

# Rename branch to main (modern convention)
git branch -M main

# Push to GitHub
git push -u origin main
```

### OR Use These Pre-filled Commands:

```bash
cd /Users/aidenm/Testch

# Set your GitHub username here:
GITHUB_USERNAME="your-username-here"

# Add remote
git remote add origin https://github.com/$GITHUB_USERNAME/infant-breathing-monitor.git

# Rename to main branch
git branch -M main

# Push
git push -u origin main
```

## Step 3: Verify

Go to your repository URL:
```
https://github.com/YOUR_USERNAME/infant-breathing-monitor
```

You should see:
- ✅ All your files
- ✅ README.md displaying
- ✅ Private repository badge
- ✅ 28 files committed

## Troubleshooting

### "Authentication failed"

You need to use a Personal Access Token (PAT) instead of password:

1. Go to: https://github.com/settings/tokens
2. Click "Generate new token" → "Generate new token (classic)"
3. Give it a name: "Infant Breathing Monitor"
4. Select scopes: ✅ **repo** (full control)
5. Click "Generate token"
6. **Copy the token** (you won't see it again!)
7. When pushing, use token as password:
   - Username: your GitHub username
   - Password: paste the token

### Alternative: Use SSH

If you have SSH keys set up:

```bash
# Use SSH URL instead
git remote add origin git@github.com:YOUR_USERNAME/infant-breathing-monitor.git
git push -u origin main
```

## What's Being Pushed

✅ **Included** (28 files):
- All Python files (5 versions)
- All documentation (12 .md files)
- Setup scripts (cross-platform)
- Configuration files
- Requirements.txt
- .gitignore

❌ **Excluded** (by .gitignore):
- venv/ (virtual environment)
- __pycache__/ (Python cache)
- *.pyc (compiled Python)
- *.csv (log files)
- screenshot_*.jpg (screenshots)
- .DS_Store (macOS)

## After Pushing

### Add a Nice README Badge

Edit your README.md and add at the top:

```markdown
# Infant Breathing Monitor AI

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Research](https://img.shields.io/badge/based%20on-MICCAI%202023-red.svg)](https://github.com/ostadabbas/Infant-Respiration-Estimation)
```

### Add Topics (Tags)

On GitHub, click "⚙️ Settings" → Add topics:
- `computer-vision`
- `opencv`
- `mediapipe`
- `infant-monitoring`
- `breathing-detection`
- `optical-flow`
- `raspberry-pi`
- `python`
- `ai`
- `healthcare`

### Add License (Optional)

If you want to add a license:

```bash
# Add MIT License
curl -o LICENSE https://raw.githubusercontent.com/licenses/license-templates/master/templates/mit.txt

# Edit with your name
# Then commit and push
git add LICENSE
git commit -m "Add MIT License"
git push
```

## Keeping It Updated

### Future Changes

```bash
# After making changes:
git add .
git commit -m "Description of changes"
git push
```

### Common Commands

```bash
# Check status
git status

# See what changed
git diff

# View commit history
git log --oneline

# Create a new branch
git checkout -b feature-name

# Push new branch
git push -u origin feature-name
```

## Share with Collaborators

To give someone access to your private repo:

1. Go to: `https://github.com/YOUR_USERNAME/infant-breathing-monitor/settings/access`
2. Click "Add people"
3. Enter their GitHub username
4. Choose permission level:
   - **Read**: Can view only
   - **Write**: Can push changes
   - **Admin**: Full control

## Repository Settings

Recommended settings (in repo Settings):

- ✅ **Issues**: Enable (for bug tracking)
- ✅ **Wiki**: Enable (for documentation)
- ✅ **Discussions**: Optional (for Q&A)
- ✅ **Automatically delete head branches**: Enable (keeps it clean)

## Need Help?

If you get stuck:
1. Check GitHub's docs: https://docs.github.com
2. GitHub's SSH setup: https://docs.github.com/en/authentication/connecting-to-github-with-ssh
3. Personal Access Token guide: https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token

---

## Quick Reference

```bash
# One-time setup
git remote add origin https://github.com/YOUR_USERNAME/infant-breathing-monitor.git
git branch -M main
git push -u origin main

# Future updates
git add .
git commit -m "Your changes"
git push
```

**Your project is ready to go! 🚀**

