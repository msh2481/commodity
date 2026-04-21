#!/usr/bin/env bash
apt update
apt install -y zsh
chsh -s $(which zsh)
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
curl -fsSL https://claude.ai/install.sh | bash

bash -c '
set -e
read -r -s -p "Anthropic API key: " KEY; echo
mkdir -p ~/.claude
echo "echo $KEY" > ~/.claude/anthropic_key.sh
chmod 700 ~/.claude/anthropic_key.sh
echo '\''{"apiKeyHelper": "~/.claude/anthropic_key.sh"}'\'' > ~/.claude/settings.json
'

cat >> ~/.zshrc <<'EOF'
alias gc="git commit -am"
alias gp="git push"
alias gu="git pull"
alias gl="git log"
alias gd="git diff"
alias ga="git add"
alias gb="git branch"
alias go="git checkout"
alias gf="git fetch"
alias gm="git merge"
alias squash="git reset --soft $(git merge-base main HEAD) && git commit -m"
EOF

echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc && source ~/.zshrc

git config --global user.name "Mikhail Budnikov"
git config --global user.email "msh24819@gmail.com"
git config --global credential.helper store
