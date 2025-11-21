---
name: feature-branch-setup
description: Automate feature branch creation following project conventions, verify branch naming patterns, check base branch state, run initial QA validation, and prepare development environment. Use when starting new features, creating bug fix branches, or setting up development branches.
---

# Feature Branch Setup

Automates the creation and setup of feature branches following Maou project conventions.

## Branch Naming Conventions

The project follows these patterns:

**Feature branches**: `feature/{topic}` or `feature/{ticket-id}-{description}`
**Bug fix branches**: `fix/{issue}` or `fix/{ticket-id}-{description}`
**Refactoring branches**: `refactor/{topic}`

Examples:
- `feature/add-array-bundling`
- `feature/gh-123-implement-mcts`
- `fix/memory-leak-training`
- `refactor/clean-architecture-domain`

## Branch Creation Workflow

### 1. Verify Current State

Check working directory is clean:

```bash
# Check for uncommitted changes
git status

# Should show: "nothing to commit, working tree clean"
```

If you have uncommitted changes, commit or stash them first:

```bash
# Stash changes
git stash

# Or commit changes
git add .
git commit -m "wip: save progress"
```

### 2. Update Base Branch

Ensure main branch is up to date:

```bash
# Switch to main branch
git checkout main

# Pull latest changes
git pull origin main

# Verify current commit
git log --oneline -1
```

### 3. Create Feature Branch

Create and switch to new branch:

```bash
# Create feature branch
git checkout -b feature/your-topic-name

# Verify branch was created
git branch --show-current
```

### 4. Run Initial QA Check

Validate that the base state passes all checks:

```bash
# Run complete QA pipeline
poetry run ruff format src/
poetry run ruff check src/ --fix
poetry run isort src/
poetry run mypy src/
poetry run pytest
```

This ensures you're starting from a clean state.

### 5. Push Branch to Remote

Set up tracking with remote:

```bash
# Push and set upstream
git push -u origin feature/your-topic-name

# Verify tracking
git branch -vv
```

## Complete Setup Script

Here's the complete workflow:

```bash
# 1. Ensure clean state
git status

# 2. Update main
git checkout main
git pull origin main

# 3. Create feature branch
git checkout -b feature/your-topic-name

# 4. Validate environment
poetry run pytest

# 5. Push to remote
git push -u origin feature/your-topic-name

# 6. Start development
echo "Branch setup complete! Ready for development."
```

## Validation Steps

### Verify Branch Name Format

Check that branch name follows conventions:

```bash
# Get current branch name
BRANCH=$(git rev-parse --abbrev-ref HEAD)

# Validate format
if [[ $BRANCH =~ ^(feature|fix|refactor)/ ]]; then
    echo "✓ Valid branch name: $BRANCH"
else
    echo "✗ Invalid branch name: $BRANCH"
    echo "Expected format: feature/*, fix/*, or refactor/*"
fi
```

### Check Base Branch

Verify you branched from main:

```bash
# Show branch relationship
git show-branch main HEAD

# Show divergence point
git merge-base main HEAD
```

### Verify Remote Tracking

Ensure branch tracks remote:

```bash
# Show tracking information
git branch -vv

# Should show: [origin/feature/your-topic-name]
```

## Development Environment Setup

### Install Dependencies

Ensure Poetry environment is ready:

```bash
# Install all dependencies
poetry install

# For specific environments
poetry install -E cpu -E gcp      # CPU + GCP
poetry install -E cuda -E aws     # CUDA + AWS
poetry install -E tpu -E gcp      # TPU + GCP
```

### Setup Pre-commit Hooks

Install pre-commit hooks for automatic validation:

```bash
poetry run bash scripts/pre-commit.sh
```

### Verify Environment

Test that tools are working:

```bash
# Check Poetry environment
poetry env info

# Verify Python version
poetry run python --version

# Test imports
poetry run python -c "import maou; print('✓ Maou package imports successfully')"
```

## Common Scenarios

### Scenario 1: New Feature from Scratch

```bash
git checkout main
git pull origin main
git checkout -b feature/new-awesome-feature
poetry run pytest  # Verify base state
git push -u origin feature/new-awesome-feature
```

### Scenario 2: Bug Fix Branch

```bash
git checkout main
git pull origin main
git checkout -b fix/training-memory-leak
poetry run pytest  # Verify base state
git push -u origin fix/training-memory-leak
```

### Scenario 3: Refactoring Branch

```bash
git checkout main
git pull origin main
git checkout -b refactor/simplify-data-pipeline
poetry run pytest  # Verify base state
git push -u origin refactor/simplify-data-pipeline
```

### Scenario 4: Branch from Specific Commit

```bash
git checkout main
git pull origin main
# Branch from specific commit
git checkout -b feature/backport-fix abc1234
poetry run pytest
git push -u origin feature/backport-fix
```

## Troubleshooting

### Issue: "Branch already exists"

```bash
# Delete local branch
git branch -D feature/old-topic

# Delete remote branch (if needed)
git push origin --delete feature/old-topic
```

### Issue: "Uncommitted changes"

```bash
# Option 1: Stash changes
git stash
git stash list

# Option 2: Commit changes
git add .
git commit -m "wip: save progress before branching"

# Option 3: Discard changes (careful!)
git reset --hard HEAD
```

### Issue: "Not up to date with origin/main"

```bash
# Fetch latest
git fetch origin

# Rebase on main
git rebase origin/main

# Or merge
git merge origin/main
```

### Issue: "Tests failing on clean branch"

```bash
# This indicates main branch has issues
# Report to team and wait for fix

# Or create branch anyway (not recommended)
git checkout -b feature/your-topic-name --no-track
```

## Branch Lifecycle

1. **Create**: Use this skill to setup branch
2. **Develop**: Make changes, commit regularly
3. **Validate**: Run QA pipeline frequently
4. **Review**: Use `pr-preparation-checks` skill
5. **Merge**: Submit PR and merge to main
6. **Cleanup**: Delete branch after merge

## Integration with Other Skills

**Before starting development**:
1. Use `feature-branch-setup` (this skill)
2. Run `qa-pipeline-automation` to verify base state

**During development**:
1. Use `architecture-validator` when adding modules
2. Use `type-safety-enforcer` when adding functions
3. Use `qa-pipeline-automation` before commits

**Before submitting PR**:
1. Use `pr-preparation-checks` for final validation

## Best Practices

- **Branch early**: Create branch before making changes
- **Name descriptively**: Branch name should explain purpose
- **Track upstream**: Always set up remote tracking
- **Validate base**: Ensure tests pass before starting
- **Commit atomically**: One logical change per commit
- **Push regularly**: Push to remote frequently

## References

- **.codex/config.yaml**: Branch naming patterns (line 17)
- **CLAUDE.md**: Atomic commits (lines 351-354)
- **AGENTS.md**: Conventional commit format (lines 231-238)
- Git branching model: Feature branch workflow
