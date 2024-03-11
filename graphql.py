"""Leetcode GraphQL API

Small script to get the question details from Leetcode using their GraphQL API.
GraphQL API details thanks to

    https://github.com/skygragon/leetcode-cli

Specifically:

    https://github.com/skygragon/leetcode-cli/blob/5245886992ceb64bfb322256b2bdc24184b36d76/lib/plugins/leetcode.js#L121-L147

TODO:
- Can we get a problem based on its number?
- Can we get the daily challenge?
"""

import requests

url = "https://leetcode.com/graphql"
headers = {
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0",
    "Connection": "keep-alive",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept": "*/*",
    "Cache-Control": "no-cache",
}


def get_question_detail(title_slug):
    opt = {
        "operationName": "getQuestionDetail",
        "variables": {"titleSlug": title_slug},
        "query": "query getQuestionDetail($titleSlug: String!) { question(titleSlug: $titleSlug) { content stats codeDefinition sampleTestCase enableRunCode metaData translatedContent } }",
    }
    res = requests.post(url, json=opt, headers=headers, timeout=10)
    return res.json()


# pretty print JSON
json = get_question_detail("two-sum")
print(list(json["data"]["question"].keys()))
print(json["data"]["question"]["content"])
