# Publishing Guide for CorpusCraft

This guide will walk you through publishing CorpusCraft to PyPI.

## Prerequisites

- GitHub repository already set up: ✅ https://github.com/belgrano9/corpuscraft
- PyPI account (create at https://pypi.org/account/register/)

## Part 1: Push to GitHub

Your repository is already configured! Just push your commits:

```bash
# Push all commits to GitHub
git push origin main

# View your repo at: https://github.com/belgrano9/corpuscraft
```

## Part 2: Publish to PyPI

### Step 1: Create PyPI Account

1. Sign up at https://pypi.org/account/register/
2. Verify your email
3. (Recommended) Set up 2FA for security

### Step 2: Generate PyPI API Token

1. Go to https://pypi.org/manage/account/token/
2. Click "Add API token"
3. Token name: `corpuscraft-upload`
4. Scope: "Entire account" (you can limit to project later)
5. **SAVE THE TOKEN** - you'll only see it once!
   - Format: `pypi-AgEIcHlwaS5vcmc...`

### Step 3: Build the Package

```bash
# Install build tools
pip install build twine

# Build the package
python -m build

# This creates:
# dist/corpuscraft-0.1.0-py3-none-any.whl
# dist/corpuscraft-0.1.0.tar.gz
```

### Step 4: Test Upload to TestPyPI (Optional but Recommended)

```bash
# Create account at https://test.pypi.org/account/register/

# Upload to TestPyPI
python -m twine upload --repository testpypi dist/*

# Test installation
pip install --index-url https://test.pypi.org/simple/ corpuscraft
```

### Step 5: Upload to PyPI

```bash
# Upload to real PyPI
python -m twine upload dist/*

# When prompted:
# Username: __token__
# Password: <paste your API token including the pypi- prefix>
```

### Step 6: Verify Publication

Visit https://pypi.org/project/corpuscraft/ to see your package!

Now anyone can install it:
```bash
pip install corpuscraft
```

## Part 3: Automate Future Releases with GitHub Actions

We've set up GitHub Actions to automatically publish to PyPI when you create a release.

### Setup GitHub Secrets

1. Go to https://github.com/belgrano9/corpuscraft/settings/secrets/actions
2. Click "New repository secret"
3. Name: `PYPI_API_TOKEN`
4. Value: (paste your PyPI API token)
5. Click "Add secret"

### Create a Release (for future versions)

When you want to publish version 0.2.0:

```bash
# 1. Update version in pyproject.toml
# version = "0.2.0"

# 2. Commit and push
git add pyproject.toml
git commit -m "Bump version to 0.2.0"
git push

# 3. Create and push tag
git tag v0.2.0
git push origin v0.2.0
```

Then on GitHub:
1. Go to https://github.com/belgrano9/corpuscraft/releases/new
2. Choose tag: `v0.2.0`
3. Release title: `v0.2.0`
4. Describe what's new
5. Click "Publish release"

The GitHub Action will automatically build and upload to PyPI! 🚀

## Quick Commands Summary

```bash
# Build package
python -m build

# Upload to PyPI
python -m twine upload dist/*

# Clean build artifacts
rm -rf dist/ build/ *.egg-info

# For future releases
git tag v0.1.1
git push origin v0.1.1
```

## Add Badges to README

Once published, add these badges (they'll work automatically):

```markdown
[![PyPI version](https://badge.fury.io/py/corpuscraft.svg)](https://pypi.org/project/corpuscraft/)
[![Downloads](https://pepy.tech/badge/corpuscraft)](https://pepy.tech/project/corpuscraft)
[![Tests](https://github.com/belgrano9/corpuscraft/workflows/Tests/badge.svg)](https://github.com/belgrano9/corpuscraft/actions)
```

## Troubleshooting

### "Package already exists"
- Increment the version in `pyproject.toml`
- Delete old `dist/` files: `rm -rf dist/`
- Rebuild: `python -m build`

### "Invalid credentials"
- Username must be exactly: `__token__` (with two underscores)
- Password is your full API token including `pypi-` prefix

### "Package name already taken"
- Check https://pypi.org/project/corpuscraft/
- If taken, choose a different name in `pyproject.toml`

## Next Steps After Publishing

1. **Push to GitHub**:
   ```bash
   git push origin main
   ```

2. **Promote Your Project**:
   - Post on Reddit: r/Python, r/MachineLearning
   - Share on Twitter/LinkedIn
   - Add to awesome-python lists

3. **Set up GitHub topics**:
   Go to https://github.com/belgrano9/corpuscraft and add topics:
   `python`, `dataset`, `synthetic-data`, `llm`, `ollama`, `docling`, `qa`, `machine-learning`

4. **Monitor**:
   - Watch for issues: https://github.com/belgrano9/corpuscraft/issues
   - Check PyPI stats: https://pypistats.org/packages/corpuscraft

Congratulations on publishing CorpusCraft! 🎉
