#!/bin/bash
# Quick script to push to GitHub after creating the repository

echo "======================================================================="
echo "🚀 Push to GitHub - Infant Breathing Monitor"
echo "======================================================================="
echo ""
echo "Prerequisites:"
echo "  1. You must have created a PRIVATE repository on GitHub"
echo "  2. Repository should be empty (no README, license, etc.)"
echo ""
echo "If you haven't created the repo yet:"
echo "  👉 Go to: https://github.com/new"
echo "  - Name: infant-breathing-monitor"
echo "  - Visibility: PRIVATE ✅"
echo "  - DO NOT initialize with anything"
echo ""
read -p "Have you created the GitHub repository? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Please create the repository first, then run this script again."
    echo "Visit: https://github.com/new"
    exit 1
fi

echo ""
read -p "Enter your GitHub username: " GITHUB_USERNAME

if [ -z "$GITHUB_USERNAME" ]; then
    echo "❌ Username cannot be empty"
    exit 1
fi

echo ""
echo "Repository URL will be:"
echo "  https://github.com/$GITHUB_USERNAME/infant-breathing-monitor"
echo ""
read -p "Is this correct? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 1
fi

echo ""
echo "Setting up remote..."

# Check if remote already exists
if git remote get-url origin &> /dev/null; then
    echo "⚠️  Remote 'origin' already exists. Removing it..."
    git remote remove origin
fi

# Add the remote
git remote add origin "https://github.com/$GITHUB_USERNAME/infant-breathing-monitor.git"
echo "✓ Remote added"

# Rename branch to main
echo ""
echo "Renaming branch to 'main'..."
git branch -M main
echo "✓ Branch renamed"

# Push to GitHub
echo ""
echo "Pushing to GitHub..."
echo ""
echo "⚠️  You'll need to authenticate:"
echo "  Username: $GITHUB_USERNAME"
echo "  Password: Your Personal Access Token (NOT your password!)"
echo ""
echo "Don't have a token? Create one here:"
echo "  👉 https://github.com/settings/tokens"
echo "  - Select: 'Generate new token (classic)'"
echo "  - Check: ✅ repo (full control)"
echo ""
read -p "Press Enter to continue..." 

git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================================="
    echo "✅ SUCCESS! Your project is now on GitHub!"
    echo "======================================================================="
    echo ""
    echo "View it at:"
    echo "  👉 https://github.com/$GITHUB_USERNAME/infant-breathing-monitor"
    echo ""
    echo "Future updates:"
    echo "  git add ."
    echo "  git commit -m \"Your changes\""
    echo "  git push"
    echo ""
else
    echo ""
    echo "======================================================================="
    echo "❌ Push failed"
    echo "======================================================================="
    echo ""
    echo "Common issues:"
    echo "  1. Repository doesn't exist - create it at https://github.com/new"
    echo "  2. Authentication failed - use Personal Access Token, not password"
    echo "  3. Wrong username - check spelling"
    echo ""
    echo "See GITHUB_SETUP.md for detailed troubleshooting"
    echo ""
fi


