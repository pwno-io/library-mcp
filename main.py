import os
import re
import sys
import glob
from typing import Any, Dict, List, Set, Optional
from dataclasses import dataclass
from datetime import datetime
import httpx
from mcp.server.fastmcp import FastMCP


@dataclass
class ContentFile:
    path: str
    meta: Dict[str, Any]
    data: str
    
    @property
    def date(self) -> Optional[datetime]:
        """Extract date from metadata or fallback to file modification time"""
        if 'date' in self.meta:
            try:
                return datetime.fromisoformat(self.meta['date'].replace('Z', '+00:00'))
            except (ValueError, TypeError):
                pass
        
        try:
            return datetime.fromtimestamp(os.path.getmtime(self.path))
        except OSError:
            return None


class HugoContentManager:
    def __init__(self, content_dirs: List[str]):
        self.content_dirs = content_dirs
        self.dir_to_files: Dict[str, List[str]] = {}
        self.path_to_content: Dict[str, ContentFile] = {}
        self.load_content()
        
    def load_content(self) -> None:
        """Load all content from the specified directories"""
        self.dir_to_files = {}
        self.path_to_content = {}
        
        for content_dir in self.content_dirs:
            if not os.path.isdir(content_dir):
                print(f"Warning: {content_dir} is not a valid directory, skipping")
                continue
                
            md_files = []
            for root, _, files in os.walk(content_dir):
                for file in files:
                    if file.endswith('.md'):
                        full_path = os.path.join(root, file)
                        md_files.append(full_path)
            
            self.dir_to_files[content_dir] = md_files
            
            for file_path in md_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    meta, data = self._parse_markdown(content)
                    self.path_to_content[file_path] = ContentFile(
                        path=file_path,
                        meta=meta,
                        data=data
                    )
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
    
    def _parse_markdown(self, content: str) -> tuple[Dict[str, Any], str]:
        """Parse markdown content, separating front matter from content"""
        front_matter_pattern = r"^---\s*\n(.*?)\n---\s*\n"
        match = re.search(front_matter_pattern, content, re.DOTALL)
        
        if not match:
            return {}, content
            
        front_matter_text = match.group(1)
        meta = {}
        
        for line in front_matter_text.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                # Handle lists in YAML (e.g., tags)
                if value.startswith('[') and value.endswith(']'):
                    value = [item.strip().strip('"\'') for item in value[1:-1].split(',')]
                
                # Handle quoted strings
                elif (value.startswith('"') and value.endswith('"')) or \
                     (value.startswith("'") and value.endswith("'")):
                    value = value[1:-1]
                    
                meta[key] = value
        
        # Extract the actual content (everything after front matter)
        data = content[match.end():]
        
        return meta, data
    
    def get_by_tag(self, tag: str, limit: int = 50) -> List[ContentFile]:
        """Find all files with a given tag"""
        matches = []
        
        for file_path, content_file in self.path_to_content.items():
            tags = content_file.meta.get('tags', [])
            
            # Handle both list and string formats for tags
            if isinstance(tags, str):
                tags = [tag.strip() for tag in tags.split(',')]
            
            if tag.lower() in [t.lower() for t in tags]:
                matches.append(content_file)
        
        # Sort by date (most recent first)
        matches.sort(key=lambda x: x.date if x.date else datetime.min, reverse=True)
        
        return matches[:limit]
    
    def get_by_text(self, query: str, limit: int = 50) -> List[ContentFile]:
        """Find all files containing the specified text"""
        matches = []
        
        for file_path, content_file in self.path_to_content.items():
            if query.lower() in content_file.data.lower():
                matches.append(content_file)
        
        # Sort by date (most recent first)
        matches.sort(key=lambda x: x.date if x.date else datetime.min, reverse=True)
        
        return matches[:limit]


def format_content_for_output(content_files: List[ContentFile]) -> str:
    """Format the content files for output"""
    result = []
    
    for file in content_files:
        result.append(f"File: {file.path}")
        result.append("Metadata:")
        for key, value in file.meta.items():
            result.append(f"  {key}: {value}")
        
        # Add a preview of the content (first 200 chars)
        preview = file.data.strip()[:200]
        if len(file.data) > 200:
            preview += "..."
            
        result.append("Content Preview:")
        result.append(f"  {preview}")
        result.append("-" * 50)
    
    if not result:
        return "No matching content found."
        
    return "\n".join(result)


# Create MCP server
mcp = FastMCP("hugo_content")
content_manager = None


@mcp.tool()
async def get_by_tag(tag: str, limit: int = 50) -> str:
    """Get blog content by its tag.
    
    Args:
        tag: the tag associated with content
        limit: the number of results to include
    """
    if content_manager is None:
        return "Content has not been loaded. Please ensure the server is properly initialized."
    
    matching_content = content_manager.get_by_tag(tag, limit)
    return format_content_for_output(matching_content)


@mcp.tool()
async def get_by_text(query: str, limit: int = 50) -> str:
    """Get blog content by text in content.
    
    Args:
        query: text for an exact match
        limit: the number of results to include
    """
    if content_manager is None:
        return "Content has not been loaded. Please ensure the server is properly initialized."
    
    matching_content = content_manager.get_by_text(query, limit)
    return format_content_for_output(matching_content)


@mcp.tool()
async def rebuild() -> bool:
    """Rebuild text index. Useful for when contents have changed on disk"""
    if content_manager is None:
        return False
    
    content_manager.load_content()
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <content_dir1> [<content_dir2> ...]")
        sys.exit(1)
    
    content_dirs = sys.argv[1:]
    print(f"Loading content from directories: {', '.join(content_dirs)}")
    
    content_manager = HugoContentManager(content_dirs)
    print(f"Loaded {len(content_manager.path_to_content)} markdown files")
    
    mcp.run(transport='stdio')