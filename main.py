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
import numpy as np

# Try to import sentence-transformers for semantic similarity
try:
    from sentence_transformers import SentenceTransformer
    SEMANTIC_SEARCH_AVAILABLE = True
except ImportError:
    SEMANTIC_SEARCH_AVAILABLE = False
    SentenceTransformer = None

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
            # Get modification time as naive datetime (assumed UTC)
            mod_time = datetime.fromtimestamp(os.path.getmtime(self.path))
            # Return as naive datetime for consistent comparison
            return mod_time
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
        
        # Initialize semantic search components
        self.semantic_model = None
        self.tag_embeddings: Dict[str, np.ndarray] = {}
        self.tag_to_text: Dict[str, str] = {}  # Store original tag text
        
        if SEMANTIC_SEARCH_AVAILABLE:
            try:
                debug_print("Initializing semantic search model...")
                self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
                debug_print("Semantic search model loaded successfully")
            except Exception as e:
                debug_print(f"Failed to load semantic model: {e}")
                self.semantic_model = None
        else:
            debug_print("Semantic search not available - install sentence-transformers for this feature")
        
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
        
        # Generate tag embeddings for semantic search
        if self.semantic_model:
            self._generate_tag_embeddings()
    
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
    
    def get_by_tags(self, tags: Union[str, List[str]], limit: int = 50) -> List[ContentFile]:
        """Find all files with any of the given tags"""
        # Normalize input to list
        if isinstance(tags, str):
            search_tags = [tags]
        else:
            search_tags = tags
            
        matches = []
        search_tags_lower = [tag.lower() for tag in search_tags]
        
        debug_print(f"Searching for tags: {search_tags}")
        for file_path, content_file in self.path_to_content.items():
            raw_tags = content_file.meta.get('tags', [])
            file_tags = self._normalize_tags(raw_tags)
            
            # Debug
            if file_tags:
                debug_print(f"File: {os.path.basename(file_path)} - Tags: {file_tags}")
            
            # Check if any of the search tags match any file tags
            matched = False
            for search_tag in search_tags_lower:
                # Check for exact tag match (case insensitive)
                if any(search_tag == t.lower() for t in file_tags):
                    debug_print(f"Found exact tag match '{search_tag}' in {os.path.basename(file_path)}")
                    matches.append(content_file)
                    matched = True
                    break
                
                # Check if the search tag is part of any file tag
                for file_tag in file_tags:
                    if search_tag in file_tag.lower():
                        debug_print(f"Found partial tag match '{search_tag}' in tag '{file_tag}' in {os.path.basename(file_path)}")
                        matches.append(content_file)
                        matched = True
                        break
                
                if matched:
                    break
        
        debug_print(f"Found {len(matches)} files with tags {search_tags}")
        
        # Sort by date (most recent first)
        def get_sort_key(content_file):
            date = content_file.date
            if date is None:
                return datetime.min
            # Make date naive if it has timezone info
            if hasattr(date, 'tzinfo') and date.tzinfo is not None:
                date = date.replace(tzinfo=None)
            return date
            
        matches.sort(key=get_sort_key, reverse=True)
        
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
        def get_sort_key(content_file):
            date = content_file.date
            if date is None:
                return datetime.min
            # Make date naive if it has timezone info
            if hasattr(date, 'tzinfo') and date.tzinfo is not None:
                date = date.replace(tzinfo=None)
            return date
            
        matches.sort(key=get_sort_key, reverse=True)
        
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
            post_date = content_file.date
            
            for tag in tags:
                if tag in tag_info:
                    count, latest_date = tag_info[tag]
                    # Handle the case where either date might be None
                    if latest_date is None:
                        new_latest = post_date
                    elif post_date is None:
                        new_latest = latest_date
                    else:
                        # Make both dates naive if they're not already
                        if hasattr(latest_date, 'tzinfo') and latest_date.tzinfo is not None:
                            latest_date = latest_date.replace(tzinfo=None)
                        if hasattr(post_date, 'tzinfo') and post_date.tzinfo is not None:
                            post_date = post_date.replace(tzinfo=None)
                        new_latest = max(latest_date, post_date)
                    tag_info[tag] = (count + 1, new_latest)
                else:
                    tag_info[tag] = (1, post_date)
        
        # Convert to list of tuples (tag, count, latest_date)
        result = [(tag, count, date) for tag, (count, date) in tag_info.items()]
        
        # Sort by count (descending) and then by date (most recent first)
        # Make all dates naive for comparison
        def get_sort_key(item):
            count = item[1]
            date = item[2]
            if date is None:
                return (-count, datetime.min)
            # Make date naive if it has timezone info
            if hasattr(date, 'tzinfo') and date.tzinfo is not None:
                date = date.replace(tzinfo=None)
            return (-count, date)
            
        result.sort(key=get_sort_key, reverse=True)
        
        debug_print(f"Collected statistics for {len(result)} tags")
        return result
    
    def _generate_tag_embeddings(self) -> None:
        """Generate embeddings for all unique tags"""
        if not self.semantic_model:
            return
            
        # Collect all unique tags
        all_tags = set()
        for _, content_file in self.path_to_content.items():
            raw_tags = content_file.meta.get('tags', [])
            tags = self._normalize_tags(raw_tags)
            all_tags.update(tags)
        
        if not all_tags:
            debug_print("No tags found to generate embeddings for")
            return
            
        debug_print(f"Generating embeddings for {len(all_tags)} unique tags...")
        
        # Convert to list for batch encoding
        tag_list = list(all_tags)
        
        try:
            # Generate embeddings in batch
            embeddings = self.semantic_model.encode(tag_list, convert_to_numpy=True)
            
            # Store embeddings
            self.tag_embeddings = {}
            self.tag_to_text = {}
            for tag, embedding in zip(tag_list, embeddings):
                self.tag_embeddings[tag] = embedding
                self.tag_to_text[tag] = tag
                
            debug_print(f"Successfully generated embeddings for {len(self.tag_embeddings)} tags")
        except Exception as e:
            debug_print(f"Error generating tag embeddings: {e}")
            self.tag_embeddings = {}
            self.tag_to_text = {}
    
    def semantic_search_tags(self, query: str, limit: int = 10, similarity_threshold: float = 0.3) -> List[Tuple[str, float]]:
        """Find semantically similar tags using embeddings"""
        if not self.semantic_model or not self.tag_embeddings:
            debug_print("Semantic search not available or no tag embeddings found")
            return []
            
        try:
            # Generate embedding for the query
            query_embedding = self.semantic_model.encode([query], convert_to_numpy=True)[0]
            
            # Calculate cosine similarity with all tags
            similarities = []
            for tag, tag_embedding in self.tag_embeddings.items():
                # Cosine similarity
                similarity = np.dot(query_embedding, tag_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(tag_embedding))
                if similarity >= similarity_threshold:
                    similarities.append((tag, float(similarity)))
            
            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            debug_print(f"Found {len(similarities)} semantically similar tags for '{query}'")
            
            return similarities[:limit]
            
        except Exception as e:
            debug_print(f"Error in semantic tag search: {e}")
            return []
    
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
            if post_date:
                # Make date naive for comparison if it has timezone info
                if hasattr(post_date, 'tzinfo') and post_date.tzinfo is not None:
                    post_date = post_date.replace(tzinfo=None)
                
                # Make start and end dates naive for comparison
                start_naive = start_date
                if hasattr(start_naive, 'tzinfo') and start_naive.tzinfo is not None:
                    start_naive = start_naive.replace(tzinfo=None)
                    
                end_naive = end_date
                if hasattr(end_naive, 'tzinfo') and end_naive.tzinfo is not None:
                    end_naive = end_naive.replace(tzinfo=None)
                
                if start_naive <= post_date <= end_naive:
                    matches.append(content_file)
        
        debug_print(f"Found {len(matches)} posts within date range")
        
        # Sort by date (most recent first)
        def get_sort_key(content_file):
            date = content_file.date
            if date is None:
                return datetime.min
            # Make date naive if it has timezone info
            if hasattr(date, 'tzinfo') and date.tzinfo is not None:
                date = date.replace(tzinfo=None)
            return date
            
        matches.sort(key=get_sort_key, reverse=True)
        
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
        if date is None:
            date_str = "Unknown"
        else:
            # Strip timezone info for display if present
            if hasattr(date, 'tzinfo') and date.tzinfo is not None:
                date = date.replace(tzinfo=None)
            
            # Only use date part for display
            if date != datetime.min:
                date_str = date.strftime("%Y-%m-%d")
            else:
                date_str = "Unknown"
                
        result.append(f"- {tag}: {count} posts, most recent: {date_str}")
    
    return "\n".join(result)


# Create MCP server
mcp = FastMCP("knowledge_base", port=5500, host="0.0.0.0")
content_manager = None


@mcp.tool()
async def get_by_tags(tags: str, limit: int = 50) -> str:
    """Get blog content by tags (supports multiple tags).
    
    Args:
        tags: a single tag or comma-separated list of tags
        limit: the number of results to include
    """
    if content_manager is None:
        return "Content has not been loaded. Please ensure the server is properly initialized."
    
    # Parse comma-separated tags
    if ',' in tags:
        tag_list = [tag.strip() for tag in tags.split(',')]
    else:
        tag_list = tags.strip()
    
    matching_content = content_manager.get_by_tags(tag_list, limit)
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
async def semantic_similar_tags(query: str, limit: int = 10, threshold: float = 0.3) -> str:
    """Find semantically similar tags AI embeddings.
    
    Args:
        query: the concept or tag to search for similar tags
        limit: the maximum number of similar tags to return
        threshold: minimum similarity score (0-1) to include a tag
    """
    if content_manager is None:
        return "Content has not been loaded. Please ensure the server is properly initialized."
    
    if not SEMANTIC_SEARCH_AVAILABLE:
        return "Semantic search is not available. Please install sentence-transformers: pip install sentence-transformers"
    
    similar_tags = content_manager.semantic_search_tags(query, limit, threshold)
    
    if not similar_tags:
        return f"No semantically similar tags found for '{query}'."
    
    result = [f"Tags semantically similar to '{query}':"]
    for tag, similarity in similar_tags:
        result.append(f"- {tag} (similarity: {similarity:.3f})")
    
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
        # Always create naive datetimes for consistent comparison
        start = datetime.fromisoformat(f"{start_date}T00:00:00")
        end = datetime.fromisoformat(f"{end_date}T23:59:59")
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
    
    mcp.run(transport='streamable-http')