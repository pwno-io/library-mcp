import os
import re
import sys
import glob
import yaml
from typing import Any, Dict, List, Set, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
import httpx
from mcp.server.fastmcp import FastMCP

# Redirect all debug prints to stderr
def debug_print(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


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
                date_str = str(self.meta['date'])
                if 'T' in date_str and not date_str.endswith('Z') and '+' not in date_str:
                    date_str += 'Z'  # Add UTC indicator if missing
                return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            except (ValueError, TypeError) as e:
                debug_print(f"Error parsing date: {e} for {self.path}")
                pass
        
        try:
            return datetime.fromtimestamp(os.path.getmtime(self.path))
        except OSError:
            return None

    @property
    def slug(self) -> str:
        """Extract slug from metadata or from filename"""
        if 'slug' in self.meta:
            return str(self.meta['slug'])
        
        # Extract from filename (basename without extension)
        filename = os.path.basename(self.path)
        return os.path.splitext(filename)[0]

    @property
    def url(self) -> Optional[str]:
        """Extract URL from metadata if available"""
        if 'url' in self.meta:
            return str(self.meta['url'])
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
                debug_print(f"Warning: {content_dir} is not a valid directory, skipping")
                continue
                
            md_files = []
            for root, _, files in os.walk(content_dir):
                for file in files:
                    if file.endswith('.md'):
                        full_path = os.path.join(root, file)
                        md_files.append(full_path)
            
            self.dir_to_files[content_dir] = md_files
            debug_print(f"Found {len(md_files)} markdown files in {content_dir}")
            
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
                    debug_print(f"Error processing {file_path}: {e}")
        
        debug_print(f"Total files processed: {len(self.path_to_content)}")
    
    def _normalize_tags(self, tags: Union[str, List, None]) -> List[str]:
        """Normalize tags to a list format regardless of input type"""
        if tags is None:
            return []
            
        if isinstance(tags, list):
            return [str(tag).strip() for tag in tags]
            
        if isinstance(tags, str):
            # If it looks like a YAML list
            if tags.startswith('[') and tags.endswith(']'):
                inner = tags[1:-1].strip()
                if not inner:
                    return []
                return [tag.strip().strip('\'"') for tag in inner.split(',')]
            
            # If it's a comma-separated string
            if ',' in tags:
                return [tag.strip() for tag in tags.split(',')]
            
            # Single tag
            return [tags.strip()]
            
        # Any other type, convert to string and return as single item
        return [str(tags)]
    
    def _parse_markdown(self, content: str) -> Tuple[Dict[str, Any], str]:
        """Parse markdown content, separating front matter from content"""
        front_matter_pattern = r"^---\s*\n(.*?)\n---\s*\n"
        match = re.search(front_matter_pattern, content, re.DOTALL)
        
        if not match:
            return {}, content
            
        front_matter_text = match.group(1)
        
        # Use PyYAML to properly parse the front matter
        try:
            meta = yaml.safe_load(front_matter_text) or {}
        except Exception as e:
            debug_print(f"YAML parsing error: {e}")
            meta = {}
        
        # Ensure meta is a dictionary
        if not isinstance(meta, dict):
            debug_print(f"Front matter did not parse as a dictionary: {type(meta)}")
            meta = {}
            
        # Ensure tags are always in list format
        if 'tags' in meta and meta['tags'] is not None:
            if not isinstance(meta['tags'], list):
                meta['tags'] = [meta['tags']]
                
        # Extract the actual content (everything after front matter)
        data = content[match.end():]
        
        return meta, data
    
    def get_by_tag(self, tag: str, limit: int = 50) -> List[ContentFile]:
        """Find all files with a given tag"""
        matches = []
        tag_lower = tag.lower()
        
        debug_print(f"Searching for tag: '{tag_lower}'")
        for file_path, content_file in self.path_to_content.items():
            raw_tags = content_file.meta.get('tags', [])
            tags = self._normalize_tags(raw_tags)
            
            # Debug
            if tags:
                debug_print(f"File: {os.path.basename(file_path)} - Tags: {tags}")
            
            # Check for exact tag match (case insensitive)
            if any(tag_lower == t.lower() for t in tags):
                debug_print(f"Found exact tag match in {os.path.basename(file_path)}")
                matches.append(content_file)
                continue
            
            # Check if the tag is part of a tag
            for t in tags:
                if tag_lower in t.lower():
                    debug_print(f"Found partial tag match in {os.path.basename(file_path)}: '{t}'")
                    matches.append(content_file)
                    break
        
        debug_print(f"Found {len(matches)} files with tag '{tag}'")
        
        # Sort by date (most recent first)
        matches.sort(key=lambda x: x.date if x.date else datetime.min, reverse=True)
        
        return matches[:limit]
    
    def get_by_text(self, query: str, limit: int = 50) -> List[ContentFile]:
        """Find all files containing the specified text"""
        matches = []
        query_lower = query.lower()
        
        debug_print(f"Searching for text: '{query}'")
        for file_path, content_file in self.path_to_content.items():
            if query_lower in content_file.data.lower():
                matches.append(content_file)
        
        debug_print(f"Found {len(matches)} files containing '{query}'")
        
        # Sort by date (most recent first)
        matches.sort(key=lambda x: x.date if x.date else datetime.min, reverse=True)
        
        return matches[:limit]

    def search_tags(self, tag_query: str, limit: int = 20) -> List[str]:
        """Search for tags matching the provided query"""
        all_tags = set()
        tag_query_lower = tag_query.lower()
        
        debug_print(f"Searching for tags containing: '{tag_query_lower}'")
        for _, content_file in self.path_to_content.items():
            raw_tags = content_file.meta.get('tags', [])
            tags = self._normalize_tags(raw_tags)
            
            # Add tags that match the query
            for tag in tags:
                if tag_query_lower in tag.lower():
                    all_tags.add(tag)
        
        # Convert to list and sort alphabetically
        tag_list = sorted(list(all_tags))
        debug_print(f"Found {len(tag_list)} tags matching '{tag_query_lower}'")
        
        return tag_list[:limit]
    
    def list_all_tags(self) -> List[Tuple[str, int, Optional[datetime]]]:
        """List all tags with their post count and most recent post date"""
        tag_info: Dict[str, Tuple[int, Optional[datetime]]] = {}
        
        debug_print("Collecting tag statistics...")
        for _, content_file in self.path_to_content.items():
            raw_tags = content_file.meta.get('tags', [])
            tags = self._normalize_tags(raw_tags)
            post_date = content_file.date or datetime.min
            
            for tag in tags:
                if tag in tag_info:
                    count, latest_date = tag_info[tag]
                    tag_info[tag] = (count + 1, max(latest_date, post_date))
                else:
                    tag_info[tag] = (1, post_date)
        
        # Convert to list of tuples (tag, count, latest_date)
        result = [(tag, count, date) for tag, (count, date) in tag_info.items()]
        
        # Sort by count (descending) and then by date (most recent first)
        result.sort(key=lambda x: (-x[1], x[2] if x[2] else datetime.min), reverse=True)
        
        debug_print(f"Collected statistics for {len(result)} tags")
        return result
    
    def get_by_slug_or_url(self, identifier: str) -> Optional[ContentFile]:
        """Find a post by its slug or URL"""
        identifier_lower = identifier.lower()
        
        debug_print(f"Searching for post with slug or URL: '{identifier}'")
        
        # First check for exact URL match (case insensitive)
        for _, content_file in self.path_to_content.items():
            url = content_file.url
            if url and url.lower() == identifier_lower:
                debug_print(f"Found exact URL match: {url}")
                return content_file
        
        # Then check for exact slug match (case insensitive)
        for _, content_file in self.path_to_content.items():
            slug = content_file.slug
            if slug.lower() == identifier_lower:
                debug_print(f"Found exact slug match: {slug}")
                return content_file
        
        # Try partial path match if no exact matches found
        for path, content_file in self.path_to_content.items():
            if identifier_lower in path.lower():
                debug_print(f"Found partial path match: {path}")
                return content_file
        
        debug_print(f"No post found for '{identifier}'")
        return None
    
    def get_by_date_range(self, start_date: datetime, end_date: datetime, limit: int = 50) -> List[ContentFile]:
        """Find all posts within a date range"""
        matches = []
        
        debug_print(f"Searching for posts between {start_date} and {end_date}")
        for _, content_file in self.path_to_content.items():
            post_date = content_file.date
            if post_date and start_date <= post_date <= end_date:
                matches.append(content_file)
        
        debug_print(f"Found {len(matches)} posts within date range")
        
        # Sort by date (most recent first)
        matches.sort(key=lambda x: x.date if x.date else datetime.min, reverse=True)
        
        return matches[:limit]


def format_content_for_output(content_files: List[ContentFile]) -> str:
    """Format the content files for output"""
    if not content_files:
        return "No matching content found."
    
    result = []
    
    for i, file in enumerate(content_files):
        result.append(f"File: {file.path}")
        result.append("Metadata:")
        for key, value in file.meta.items():
            result.append(f"  {key}: {value}")
        
        # Include the full content
        result.append("Content:")
        result.append(file.data.strip())
        
        # Add separator between entries, but not after the last one
        if i < len(content_files) - 1:
            result.append("-" * 50)
    
    return "\n".join(result)


def format_tags_for_output(tags: List[Tuple[str, int, Optional[datetime]]]) -> str:
    """Format tag information for output"""
    if not tags:
        return "No tags found."
    
    result = []
    result.append("Tags (by post count and most recent post):")
    
    for tag, count, date in tags:
        date_str = date.strftime("%Y-%m-%d") if date and date != datetime.min else "Unknown"
        result.append(f"- {tag}: {count} posts, most recent: {date_str}")
    
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
    
    debug_print("Rebuilding content index...")
    content_manager.load_content()
    debug_print("Content index rebuilt successfully")
    return True


@mcp.tool()
async def search_tags(tag_query: str, limit: int = 20) -> str:
    """Search for tags matching the provided query.
    
    Args:
        tag_query: partial or full tag name to search for
        limit: the maximum number of tags to return
    """
    if content_manager is None:
        return "Content has not been loaded. Please ensure the server is properly initialized."
    
    matching_tags = content_manager.search_tags(tag_query, limit)
    
    if not matching_tags:
        return f"No tags found matching '{tag_query}'."
    
    result = [f"Tags matching '{tag_query}':"]
    for tag in matching_tags:
        result.append(f"- {tag}")
    
    return "\n".join(result)


@mcp.tool()
async def list_all_tags() -> str:
    """List all tags sorted by number of posts and most recent post."""
    if content_manager is None:
        return "Content has not been loaded. Please ensure the server is properly initialized."
    
    tag_info = content_manager.list_all_tags()
    return format_tags_for_output(tag_info)


@mcp.tool()
async def get_by_slug_or_url(identifier: str) -> str:
    """Get a post by its slug or URL.
    
    Args:
        identifier: the slug, URL, or path fragment to search for
    """
    if content_manager is None:
        return "Content has not been loaded. Please ensure the server is properly initialized."
    
    post = content_manager.get_by_slug_or_url(identifier)
    
    if post is None:
        return f"No post found with slug or URL matching '{identifier}'."
    
    # Format as a list to reuse format_content_for_output
    return format_content_for_output([post])


@mcp.tool()
async def get_by_date_range(start_date: str, end_date: str, limit: int = 50) -> str:
    """Get posts published within a date range.
    
    Args:
        start_date: the start date in ISO format (YYYY-MM-DD)
        end_date: the end date in ISO format (YYYY-MM-DD)
        limit: the maximum number of posts to return
    """
    if content_manager is None:
        return "Content has not been loaded. Please ensure the server is properly initialized."
    
    try:
        # Parse dates with time set to beginning/end of day
        start = datetime.fromisoformat(f"{start_date}T00:00:00+00:00")
        end = datetime.fromisoformat(f"{end_date}T23:59:59+00:00")
    except ValueError as e:
        return f"Error parsing dates: {e}. Please use ISO format (YYYY-MM-DD)."
    
    posts = content_manager.get_by_date_range(start, end, limit)
    return format_content_for_output(posts)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        debug_print("Usage: python script.py <content_dir1> [<content_dir2> ...]")
        sys.exit(1)
    
    content_dirs = sys.argv[1:]
    debug_print(f"Loading content from directories: {', '.join(content_dirs)}")
    
    content_manager = HugoContentManager(content_dirs)
    debug_print(f"Loaded {len(content_manager.path_to_content)} markdown files")
    
    mcp.run(transport='stdio')