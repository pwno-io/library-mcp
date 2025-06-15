[library-mcp](https://github.com/lethain/library-mcp) is an MCP server for interacting with
Markdown knowledge bases. Basically, folders that may or may not have subfolders, that include
files with `.md` extension and start with metadata like:

    ----
    title: My blog post
    tags:
    - python
    - programming
    url: /my-blog-post
    ---

    # My blog post
    Yesterday I was dreaming about...

The typical workflow in the current verison is to
retrieve recent content for a given tag or tags, and then
discuss using those tags:

    Get the next 50 posts with tags "executive" or "management",
    then tell me what I should do about this problem
    I am running into: ...

You can also do the same but by date ranges:

    Summarize the blog posts I wrote in the past year.

You might reasonably ask "why not just upload your entire blog
into the context window?" and there are two places where this library
outperforms that approach:

1. My blog corpus is much larger than most model's context windows today.
    Further, even if the context windows became exhaustively large, I wrote a lot
    of mediocre stuff in the past, so maybe omitting it's a feature.
2. I have a number of distinct Markdown knowledge bases, and this lets me
    operate across them in tandem.

Finally, this is a hobby project, intended for running locally on your
laptop. No humans have been harmed using this software, but it does
work pretty well!


# Tools

This MCP server exposes these tools.

### Content Search Tools

Tools for retrieving content into your context window:

* `get_by_tags` - Retrieves content by one or more tags (comma-separated)
* `get_by_text` - Searches content for specific text
* `get_by_slug_or_url` - Finds posts by slug or URL
* `get_by_date_range` - Gets posts published within a date range

### Tag Management Tools

Tools for navigating your knowledge base:

* `search_tags` - Searches for tags matching a query
* `list_all_tags` - Lists all tags sorted by post count and recency
* `semantic_similar_tags` - Finds conceptually similar tags using AI embeddings (requires optional dependencies)

### Maintenance Tools

Tools for dealing with running the tool:

* `rebuild` - Rebuilds the content index,
    useful if you have added more content,
    edited existing content, etc


# Setup / Installation

These instructions describe installation for [Claude Desktop](https://claude.ai/download) on OS X.
It should work similarly on other platforms.

1. Install [Claude Desktop](https://claude.ai/download).
2. Clone [library-mcp](https://github.com/lethain/library-mcp) into
    a convenient location, I'm assuming `/Users/will/library-mcp`
3. Make sure you have `uv` installed, you can [follow these instructions](https://modelcontextprotocol.io/quickstart/server)
4. Go to Cladue Desktop, Setting, Developer, and have it create your MCP config file.
    Then you want to update your `claude_desktop_config.json`.
    (Note that you should replace `will` with your user, e.g. the output of `whoami`.

        cd /Users/will/Library/Application Support/Claude
        vi claude_desktop_config.json

    Then add this section:

        {
          "mcpServers": {
            "library": {
              "command": "uv",
              "args": [
                "--directory",
                "/Users/will/library-mcp",
                "run",
                "main.py",
                "/Users/will/irrational_hugo/content"
              ]
            }
          }
        }

5. Close Claude and reopen it.
6. It should work...

## Optional: Semantic Search

To enable semantic tag search functionality, install the optional dependencies:

```bash
cd /Users/will/library-mcp
uv pip install ".[semantic]"
```

This will install `sentence-transformers` which enables the `semantic_similar_tags` tool.
This tool uses AI embeddings to find tags that are conceptually similar, even if they
don't share the same text. For example, searching for "programming" might find tags
like "coding", "development", or "software engineering".
