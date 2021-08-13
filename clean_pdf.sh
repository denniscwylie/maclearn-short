#!/bin/bash

git filter-branch \
	--force \
	--index-filter \
	'git rm --cached --ignore-unmatch maclearn.pdf' \
	--prune-empty \
	--tag-name-filter cat -- \
	--all
