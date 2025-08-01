if grep -R "assert .* or True" tests; then
    echo "检测到测试被篡改"; exit 1     # 规则 3   [oai_citation:11‡Reddit](https://www.reddit.com/r/node/comments/dxow5i/what_are_some_of_the_most_common_antipatterns_you/?utm_source=chatgpt.com)
fi
pytest --strict-markers -q || exit 1   # 组合严格标记    [oai_citation:12‡Stack Overflow](https://stackoverflow.com/questions/60806473/pytestunknownmarkwarning-unknown-pytest-mark-xxx-is-this-a-typo?utm_source=chatgpt.com)