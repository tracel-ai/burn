#!/usr/bin/env bash
root="$(git rev-parse --show-toplevel)"
dot_git="$(git rev-parse --git-common-dir)"
if [[ ! -d "${dot_git}/hooks" ]]; then mkdir -p "${dot_git}/hooks"; fi
# fix old bug
rm -f "${dot_git}/hooks/comit-msg"
rm -f "${dot_git}/hooks/commit-msg"
ln -sf "${root}/misc/git-hooks/commit-msg" "${dot_git}/hooks/commit-msg"

root="$(git rev-parse --show-toplevel)"
dot_git="$(git rev-parse --git-common-dir)"
if [[ ! -d "${dot_git}/hooks" ]]; then mkdir -p "${dot_git}/hooks"; fi
# fix old bug
rm -f "${dot_git}/hooks/pre-comit"
rm -f "${dot_git}/hooks/pre-commit"
ln -sf "${root}/misc/git-hooks/pre-commit" "${dot_git}/hooks/pre-commit"

# set template
git config commit.template misc/git-hooks/commit-template.txt

if ! flakebox lint --silent; then
  >&2 echo "ℹ️  Project recommendations detected. Run 'flakebox lint' for more info."
fi

if [ -n "${DIRENV_IN_ENVRC:-}" ]; then
  # and not set DIRENV_LOG_FORMAT
  if [ -n "${DIRENV_LOG_FORMAT:-}" ]; then
    >&2 echo "💡 Set 'DIRENV_LOG_FORMAT=\"\"' in your shell environment variables for a cleaner output of direnv"
  fi
fi

>&2 echo "💡 Run 'just' for a list of available 'just ...' helper recipes"
