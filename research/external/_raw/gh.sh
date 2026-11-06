#!/bin/bash
set -u
REPOS="sdv-dev/SDV vanderschaarlab/synthcity sdv-dev/SDGym pdebench/PDEBench AI4Science-WestlakeU/RealPDEBench sdv-dev/sdmetrics sdv-dev/Copulas realpdebench/RealPDEBench"
for r in $REPOS; do
  f=$(echo "$r" | tr '/' '_')
  code=$(curl -s -m 25 -o "$f.repo.json" -w "%{http_code}" "https://api.github.com/repos/$r")
  echo "== $r http=$code"
  if [ "$code" = "200" ]; then
    jq -r '"full_name=\(.full_name) default=\(.default_branch) stars=\(.stargazers_count) archived=\(.archived) pushed_at=\(.pushed_at) created=\(.created_at) license=\(.license.spdx_id // "NONE") lang=\(.language) desc=\(.description)"' "$f.repo.json"
    ccode=$(curl -s -m 25 -o "$f.commits.json" -w "%{http_code}" "https://api.github.com/repos/$r/commits?per_page=1")
    if [ "$ccode" = "200" ]; then
      jq -r '.[0] | "  last_commit=\(.commit.committer.date) sha=\(.sha[0:8]) msg=\(.commit.message | split("\n")[0][0:90])"' "$f.commits.json"
    else
      echo "  commits http=$ccode"
    fi
  else
    jq -r '"message=\(.message)"' "$f.repo.json" 2>/dev/null || head -c 200 "$f.repo.json"
  fi
done
