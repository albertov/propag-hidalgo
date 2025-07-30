# Git Commit Message

Generate a well-structured commit message based on staged changes and commit them

## Description
This command analyzes staged git changes and creates a clear, concise commit message following conventional commit standards. It reviews the changes, categorizes them, and suggests an appropriate commit message without actually committing and stops. When the user approves it commits. When not, iterate with the user
until a satisfactory commit message produced.


## Usage
`commit`

## Variables
- COMMIT_STYLE: Commit message format (default: conventional)
- INCLUDE_BODY: Add detailed body to message (default: true for complex changes)


## Steps
1. Run `git status` to see staged files
2. Review changes with `git diff --cached` if needed
3. Analyze the nature of changes:
   - Feature additions (feat:)
   - Bug fixes (fix:)
   - Documentation (docs:)
   - Style changes (style:)
   - Refactoring (refactor:)
   - Tests (test:)
   - Chores (chore:)
4. Write a commit message with:
   - Type and scope in subject line
   - Clear, imperative mood description
   - Body with "why" and "what" if needed
   - Footer with references if applicable
5. Show the message to the user for review and STOP!
6. Commit or iterate until the user likes the message or instructs you to stop.

## Examples
### Example 1: Simple feature commit
```
feat(auth): add password reset functionality

- Implement forgot password flow
- Add email notification service
- Create password reset tokens with 24h expiry
```

### Example 2: Bug fix commit
```
fix(api): resolve null pointer in user validation

Validation was failing when optional fields were undefined.
Added null checks before accessing nested properties.

Fixes #123
```

## Notes
- Do not add or remove any changes. Focus on the staged changes only.
- Keep subject line under 50 characters
- Use present tense ("add" not "added")
- Reference issues/PRs when relevant
