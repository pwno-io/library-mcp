



## Adding to Claude Desktop

On OS X, you can configure by going 


will@Wills-MacBook-Pro Claude % cat claude_desktop_config.json | pbcopy
will@Wills-MacBook-Pro Claude % pwd
/Users/will/Library/Application Support/Claude



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
