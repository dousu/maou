---
name: gh-pr
description: Create comprehensive GitHub pull requests using gh command with intelligently generated PR descriptions. Analyzes commit history, code changes, and project context to generate detailed PR bodies that include problem statement, solution approach, impact analysis, testing strategy, and reviewer guidance. Use when creating pull requests that require thorough documentation for effective code review.
---

# GitHub Pull Request Creation

Automates pull request creation with comprehensive, context-aware PR descriptions that provide reviewers with all necessary information for effective code review.

## Core Workflow

### 1. Pre-PR Validation

Before creating a PR, ensure all requirements are met:

```bash
# Run complete QA pipeline
uv run ruff format src/
uv run ruff check src/ --fix
uv run isort src/
uv run mypy src/
uv run pytest

# Verify working directory is clean
git status

# Ensure all changes are committed
git log --oneline -5
```

If validation fails, address issues before proceeding.

### 2. Analyze Changes Comprehensively

**CRITICAL**: Thoroughly analyze the branch to understand all changes:

```bash
# Get base branch (usually main)
BASE_BRANCH="main"

# View all commits in this branch
git log ${BASE_BRANCH}..HEAD --oneline

# Analyze complete diff from base
git diff ${BASE_BRANCH}...HEAD

# Understand file-level changes
git diff ${BASE_BRANCH}...HEAD --stat

# Check for new or deleted files
git diff ${BASE_BRANCH}...HEAD --name-status
```

**Key Analysis Points**:
- What problem does this change solve?
- What approach was taken?
- Which files were modified and why?
- Are there any breaking changes?
- What are the performance implications?
- Are there any architectural changes?
- What edge cases need reviewer attention?

### 3. Generate Comprehensive PR Body

Based on the analysis, generate a detailed PR description that includes:

#### Required Sections

**## Summary**
- 2-4 bullet points highlighting the main changes
- Focus on WHAT changed and WHY
- Use clear, concise language

**## Problem**
- Describe the issue being solved or feature being added
- Include context: Why is this change necessary?
- Reference related issues/tickets if applicable

**## Solution**
- Explain the approach taken
- Highlight key technical decisions
- Describe the implementation strategy
- Mention any trade-offs or alternatives considered

**## Changes**
- List modified components/modules
- Explain major code changes
- Note any refactoring or cleanup
- Highlight new dependencies or tools introduced

**## Impact**
- **Breaking Changes**: List any breaking changes (CRITICAL)
- **Performance**: Note performance implications (positive or negative)
- **Compatibility**: Mention compatibility considerations
- **Dependencies**: Note dependency updates or additions
- **Configuration**: List configuration changes required

**## Testing**
- **New Tests**: List new test files/cases added
- **Test Coverage**: Mention coverage improvements
- **Manual Testing**: Describe manual testing performed
- **Edge Cases**: List edge cases tested
- **Cloud Integration**: Note cloud provider testing if applicable

**## Technical Details**

Provide implementation details that help reviewers:
- Architecture changes or considerations
- Design patterns used
- Algorithm choices
- Data structure decisions
- Performance optimizations
- Error handling approach

**## Review Notes**

Guide reviewers on what to focus on:
- Areas requiring careful review
- Known limitations or caveats
- Follow-up work needed
- Specific feedback requested

**## Checklist**
```markdown
- [ ] Code formatted and linted (ruff, isort)
- [ ] Type checking passes (mypy)
- [ ] All tests pass (pytest)
- [ ] New tests added for new features
- [ ] Regression tests added for bug fixes
- [ ] Architecture compliance verified
- [ ] Docstrings added/updated
- [ ] CLAUDE.md updated if needed
- [ ] No Co-authored-by trailers
- [ ] No AI tool references
```

### 4. Create Pull Request

Use `gh pr create` with the generated body:

```bash
# Interactive mode (recommended for first-time)
gh pr create --title "TYPE(scope): brief description" --body "$(cat <<'EOF'
## Summary
- [Generated summary points]

## Problem
[Problem description]

## Solution
[Solution explanation]

## Changes
- [Change 1]
- [Change 2]

## Impact
**Breaking Changes**: None / [List if any]
**Performance**: [Performance notes]
**Dependencies**: [Dependency updates]

## Testing
- [New tests]
- [Coverage info]
- [Manual testing]

## Technical Details
[Implementation details]

## Review Notes
[Reviewer guidance]

## Checklist
- [x] Code formatted and linted
- [x] Type checking passes
- [x] All tests pass
- [x] Tests added
- [x] Architecture compliant
- [x] Docstrings updated
EOF
)"

# Specify base branch explicitly if needed
gh pr create --base main --title "..." --body "..."

# Add reviewers
gh pr create --title "..." --body "..." --reviewer username1,username2

# Add labels
gh pr create --title "..." --body "..." --label "feature,needs-review"
```

### 5. Verify PR Creation

After creation, verify the PR:

```bash
# View created PR
gh pr view

# Check PR status
gh pr status

# View PR in browser
gh pr view --web
```

## PR Title Format

Follow conventional commit format:

```
<type>(<scope>): <description>
```

**Valid types**:
- `feat`: New feature
- `fix`: Bug fix
- `refactor`: Code refactoring
- `perf`: Performance improvement
- `test`: Test additions/improvements
- `docs`: Documentation updates
- `build`: Build system changes
- `ci`: CI/CD changes

**Examples**:
```
feat(cloud): add array bundling for S3 downloads
fix(training): resolve memory leak in DataLoader
refactor(domain): simplify loss function architecture
perf(preprocessing): optimize batch processing speed
test(cloud): add integration tests for GCS operations
```

## Intelligent Content Generation

**IMPORTANT**: Do NOT use generic templates. Generate specific, accurate content by:

1. **Reading the Code Changes**
   - Actually analyze the diff
   - Understand what each file change does
   - Identify the core logic changes

2. **Understanding the Context**
   - Read related files if needed
   - Check the architecture being followed
   - Review existing patterns in the codebase

3. **Analyzing Commit Messages**
   - Extract information from commit history
   - Identify the progression of changes
   - Understand the development approach

4. **Generating Specific Content**
   - Be precise about what changed
   - Explain WHY changes were made
   - Provide concrete examples
   - Mention specific functions/classes affected

5. **Anticipating Reviewer Questions**
   - What might be unclear?
   - What decisions need justification?
   - What risks should reviewers watch for?

## Complete Example Workflow

```bash
# 1. Analyze changes comprehensively
echo "=== Analyzing branch changes ==="
git log main..HEAD --oneline
git diff main...HEAD --stat

# 2. Review detailed diff
git diff main...HEAD | less

# 3. Generate PR body (Claude Code does this intelligently)
# [Analyze code, understand context, generate specific content]

# 4. Create PR with comprehensive body
gh pr create --title "feat(cloud): add array bundling for efficient S3 caching" --body "$(cat <<'EOF'
## Summary
- Implement ~1GB array bundling to reduce S3 file count and improve I/O performance
- Add metadata management for fast array lookup within bundles
- Use memory mapping for efficient data access during training
- Support configurable bundle sizes via CLI options

## Problem
When downloading thousands of small numpy arrays from S3 for training，each file requires a separate S3 request，leading to:
- High I/O overhead (thousands of small files)
- Inefficient caching (difficult to manage many small files)
- Slower training startup times
- Increased S3 API costs

## Solution
Introduced an array bundling system that:
1. Downloads individual arrays from S3
2. Combines them into ~1GB bundles locally
3. Creates metadata files for fast lookup
4. Uses memory mapping for efficient access

Key technical decisions:
- Bundle size defaults to 1GB (configurable)
- Uses numpy.lib.format for efficient serialization
- Metadata stored in JSON for quick parsing
- Transparent to DataLoader (maintains same interface)

## Changes
- **src/maou/infra/cloud/s3_bundler.py**: New bundling logic
- **src/maou/infra/datasource/s3_datasource.py**: Integrated bundling support
- **src/maou/interface/cli/commands/learn.py**: Added CLI options
- **src/maou/interface/cli/commands/preprocess.py**: Added CLI options
- **tests/infra/cloud/test_s3_bundler.py**: Comprehensive test coverage

## Impact
**Breaking Changes**: None (opt-in feature via --input-enable-bundling flag)

**Performance**:
- 60-80% reduction in S3 download time for typical datasets
- ~50% faster training startup with cached bundles
- Reduced S3 API costs (fewer requests)

**Dependencies**: No new dependencies (uses existing numpy capabilities)

**Configuration**: New CLI options:
- --input-enable-bundling: Enable bundling feature
- --input-bundle-size-gb: Configure bundle size (default: 1.0)

## Testing
**New Tests**:
- tests/infra/cloud/test_s3_bundler.py: 15 test cases
- Test bundle creation，metadata management，and error handling
- Test memory mapping and data retrieval
- Test edge cases (empty arrays，large arrays)

**Coverage**: Added 180 lines，95% coverage

**Manual Testing**:
- Tested with 10K array dataset (10GB total)
- Verified training performance improvement
- Tested with various bundle sizes (0.5GB，1GB，2GB)
- Validated memory usage patterns

**Cloud Integration**:
- Tested with real S3 bucket
- Verified AWS credentials handling
- Tested network failure scenarios

## Technical Details
**Architecture**: Maintains Clean Architecture principles
- Bundler in infrastructure layer (src/maou/infra/cloud/)
- No domain/app layer changes needed
- Interface layer exposes CLI options

**Bundle Format**:
```
bundle_000.npy  # Combined numpy arrays
bundle_000.meta.json  # Metadata: {"array_id": {"offset": 0, "shape": [...]}}
```

**Memory Mapping**: Uses numpy.load(mmap_mode='r') for zero-copy access

**Error Handling**:
- Graceful fallback to non-bundled mode on errors
- Validates bundle integrity before use
- Logs detailed error information

## Review Notes
**Focus Areas**:
1. Bundle format and metadata structure (lines 45-89 in s3_bundler.py)
2. Memory mapping usage (lines 120-145)
3. Error handling in edge cases (lines 200-230)

**Known Limitations**:
- Bundle size is approximate (may exceed slightly)
- No automatic bundle re-creation (manual cleanup needed)

**Follow-up Work**:
- Add GCS bundling support (separate PR)
- Implement automatic bundle cleanup
- Add bundle size optimization based on dataset characteristics

## Checklist
- [x] Code formatted and linted (ruff，isort)
- [x] Type checking passes (mypy)
- [x] All tests pass (pytest)
- [x] New tests added (15 test cases)
- [x] Architecture compliance verified
- [x] Docstrings added (all public APIs)
- [x] CLAUDE.md updated (array bundling section added)
- [x] No Co-authored-by trailers
- [x] No AI tool references
EOF
)"
```

## Advanced Options

### Draft PRs

Create draft PR for early feedback:

```bash
gh pr create --draft --title "..." --body "..."
```

### PR to Different Base

Target specific base branch:

```bash
gh pr create --base develop --title "..." --body "..."
```

### Auto-merge Setup

Enable auto-merge when checks pass:

```bash
gh pr create --title "..." --body "..."
gh pr merge --auto --squash
```

## Strict Prohibitions

These will cause PR rejection:

❌ **Generic PR descriptions** - Be specific about changes
❌ **Missing impact analysis** - Always analyze breaking changes
❌ **No testing information** - Document all testing performed
❌ **Co-authored-by trailers** - Never include in commits
❌ **AI tool references** - No mentions of Claude, GPT, etc.
❌ **Incomplete checklists** - Verify all items before submission

## Integration with Project Standards

### Clean Architecture Compliance

Verify your changes respect dependency flow:
```
infra → interface → app → domain
```

Reference: CLAUDE.md lines 52-60

### Type Safety

All new code must have type hints:
```python
def process_bundle(arrays: list[np.ndarray], size_gb: float) -> Path:
    ...
```

Reference: CLAUDE.md lines 33-35

### Japanese Documentation

Use correct punctuation in Japanese text:
- 句点: ，（全角コンマ）
- 読点: ．（全角ピリオド）

Reference: CLAUDE.md lines 382-405

## Troubleshooting

### Issue: "No commits between base and HEAD"

```bash
# Verify you have commits
git log main..HEAD

# If empty, make sure you're on the right branch
git branch --show-current

# Check if you need to push commits first
git status
```

### Issue: "Authentication failed"

```bash
# Login to GitHub CLI
gh auth login

# Verify authentication
gh auth status
```

### Issue: "Base branch not found"

```bash
# Fetch latest branches
git fetch origin

# Specify base branch explicitly
gh pr create --base main --title "..." --body "..."
```

### Issue: "PR body too long"

If the PR body is very long (>65536 characters):

```bash
# Save body to file
cat > pr_body.md <<'EOF'
[Your PR body content]
EOF

# Create PR using file
gh pr create --title "..." --body-file pr_body.md
```

## Best Practices

1. **Analyze First, Write Second**: Thoroughly understand changes before writing description
2. **Be Specific**: Use concrete examples and specific file/function names
3. **Anticipate Questions**: Address potential reviewer concerns proactively
4. **Provide Context**: Explain WHY，not just WHAT
5. **Guide Review**: Tell reviewers what to focus on
6. **Complete Checklist**: Verify all items before creating PR
7. **Update Documentation**: Keep CLAUDE.md and other docs in sync

## References

- **CLAUDE.md**: PR guidelines (lines 323-336)
- **AGENTS.md**: Commit message format (lines 231-238)
- **GitHub CLI Docs**: https://cli.github.com/manual/gh_pr_create
- **Conventional Commits**: https://www.conventionalcommits.org/
