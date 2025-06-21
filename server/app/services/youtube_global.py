import os
import json
import requests
import re
from typing import List, Dict, Optional
from urllib.parse import quote_plus, urlencode
import time
import random
from bs4 import BeautifulSoup
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    GOOGLE_API_AVAILABLE = True
except ImportError:
    GOOGLE_API_AVAILABLE = False
    logger.warning("Google API client not available. Only web scraping method will work.")

class YouTubeSearchService:
    """
    A service class to search YouTube videos using multiple methods:
    1. YouTube Data API v3 (requires valid API key)
    2. Web scraping (backup method)
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the YouTube search service
        
        Args:
            api_key (str, optional): YouTube Data API v3 key
        """
        self.api_key = api_key
        self.youtube = None
        
        if api_key and GOOGLE_API_AVAILABLE:
            try:
                self.youtube = build('youtube', 'v3', developerKey=api_key)
                logger.info("YouTube API client initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize YouTube API client: {e}")
        
        # Headers for web scraping
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def _fix_api_permissions(self) -> str:
        """
        Provide instructions to fix API permission issues
        """
        return """
        API Permission Issues - Troubleshooting Steps:
        
        1. Check API Key Configuration:
           - Go to Google Cloud Console (https://console.cloud.google.com/)
           - Select your project
           - Go to APIs & Services > Credentials
           - Verify your API key is correct
        
        2. Enable YouTube Data API v3:
           - Go to APIs & Services > Library
           - Search for "YouTube Data API v3"
           - Click on it and ensure it's ENABLED
        
        3. Check API Restrictions:
           - In Credentials, click on your API key
           - Under "API restrictions", make sure YouTube Data API v3 is allowed
           - Under "Application restrictions", configure as needed
        
        4. Billing Account:
           - Ensure your Google Cloud project has billing enabled
           - YouTube API requires a billing account even for free tier usage
        
        5. Quota Limits:
           - Check if you've exceeded daily quota limits
           - Go to APIs & Services > Quotas to check usage
        
        Using web scraping method as fallback...
        """
    
    def search_videos_web_scraping(self, 
                                  query: str, 
                                  max_results: int = 50) -> Dict:
        """
        Search YouTube videos using web scraping (fallback method)
        
        Args:
            query (str): Search query
            max_results (int): Maximum number of results to return
        
        Returns:
            Dict: Search results with video information
        """
        try:
            # Encode the search query
            encoded_query = quote_plus(query)
            
            # YouTube search URL
            url = f"https://www.youtube.com/results?search_query={encoded_query}"
            
            # Add random delay to avoid rate limiting
            time.sleep(random.uniform(1, 2))
            
            # Make request
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            # Parse the HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find script tags containing video data
            scripts = soup.find_all('script')
            video_data = []
            
            for script in scripts:
                if script.string and 'var ytInitialData' in script.string:
                    # Extract JSON data
                    script_content = script.string
                    start = script_content.find('var ytInitialData = ') + len('var ytInitialData = ')
                    end = script_content.find(';</script>', start)
                    if end == -1:
                        end = script_content.find(';', start)
                    
                    json_str = script_content[start:end]
                    try:
                        data = json.loads(json_str)
                        video_data = self._extract_videos_from_json(data, max_results)
                        break
                    except json.JSONDecodeError:
                        continue
            
            # If JSON parsing failed, try regex approach
            if not video_data:
                video_data = self._extract_videos_with_regex(response.text, max_results)
            
            result = {
                'total_results': len(video_data),
                'results_per_page': len(video_data),
                'videos': video_data[:max_results],
                'query': query,
                'method': 'web_scraping'
            }
            
            logger.info(f"Successfully retrieved {len(video_data)} videos for query: '{query}' using web scraping")
            return result
            
        except Exception as e:
            logger.error(f"Web scraping error: {e}")
            return {'error': f'Web scraping error: {e}', 'videos': []}
    
    def _extract_videos_from_json(self, data: dict, max_results: int) -> List[Dict]:
        """Extract video information from YouTube's JSON data"""
        videos = []
        
        try:
            # Navigate through the complex JSON structure
            contents = data.get('contents', {}).get('twoColumnSearchResultsRenderer', {}).get('primaryContents', {}).get('sectionListRenderer', {}).get('contents', [])
            
            for content in contents:
                if 'itemSectionRenderer' in content:
                    items = content['itemSectionRenderer'].get('contents', [])
                    
                    for item in items:
                        if 'videoRenderer' in item:
                            video = item['videoRenderer']
                            
                            video_id = video.get('videoId', '')
                            title = video.get('title', {}).get('runs', [{}])[0].get('text', 'N/A')
                            
                            # Extract channel info
                            channel_info = video.get('ownerText', {}).get('runs', [{}])[0]
                            channel_title = channel_info.get('text', 'N/A')
                            
                            # Extract description
                            description_snippets = video.get('detailedMetadataSnippets', [])
                            description = ''
                            if description_snippets:
                                desc_runs = description_snippets[0].get('snippetText', {}).get('runs', [])
                                description = ' '.join([run.get('text', '') for run in desc_runs])
                            
                            # Extract thumbnail
                            thumbnails = video.get('thumbnail', {}).get('thumbnails', [])
                            thumbnail_url = thumbnails[0]['url'] if thumbnails else ''
                            
                            # Extract publish time
                            publish_time = video.get('publishedTimeText', {}).get('simpleText', 'N/A')
                            
                            video_info = {
                                'video_id': video_id,
                                'title': title,
                                'description': description,
                                'channel_title': channel_title,
                                'channel_id': 'N/A',  # Not easily available in web scraping
                                'published_at': publish_time,
                                'thumbnail_url': thumbnail_url,
                                'video_url': f"https://www.youtube.com/watch?v={video_id}"
                            }
                            
                            videos.append(video_info)
                            
                            if len(videos) >= max_results:
                                break
                
                if len(videos) >= max_results:
                    break
        
        except Exception as e:
            logger.error(f"Error extracting videos from JSON: {e}")
        
        return videos
    
    def _extract_videos_with_regex(self, html_content: str, max_results: int) -> List[Dict]:
        """Fallback method using regex to extract video information"""
        videos = []
        
        try:
            # Regex patterns for video data
            video_id_pattern = r'"videoId":"([^"]+)"'
            title_pattern = r'"title":{"runs":\[{"text":"([^"]+)"'
            channel_pattern = r'"ownerText":{"runs":\[{"text":"([^"]+)"'
            
            video_ids = re.findall(video_id_pattern, html_content)
            titles = re.findall(title_pattern, html_content)
            channels = re.findall(channel_pattern, html_content)
            
            # Combine the extracted data
            for i in range(min(len(video_ids), len(titles), max_results)):
                video_info = {
                    'video_id': video_ids[i] if i < len(video_ids) else 'N/A',
                    'title': titles[i] if i < len(titles) else 'N/A',
                    'description': 'N/A',
                    'channel_title': channels[i] if i < len(channels) else 'N/A',
                    'channel_id': 'N/A',
                    'published_at': 'N/A',
                    'thumbnail_url': f"https://img.youtube.com/vi/{video_ids[i]}/default.jpg" if i < len(video_ids) else '',
                    'video_url': f"https://www.youtube.com/watch?v={video_ids[i]}" if i < len(video_ids) else ''
                }
                videos.append(video_info)
        
        except Exception as e:
            logger.error(f"Error in regex extraction: {e}")
        
        return videos
    
    def search_videos(self, 
                     query: str, 
                     max_results: int = 2,
                     order: str = 'relevance',
                     published_after: Optional[str] = None,
                     published_before: Optional[str] = None,
                     video_duration: Optional[str] = None,
                     video_definition: Optional[str] = None,
                     use_web_scraping: bool = False) -> Dict:
        """
        Search for YouTube videos using API or web scraping
        
        Args:
            query (str): Search query (can be tags, title, or topics)
            max_results (int): Maximum number of results to return (default: 50, max: 50)
            order (str): Order of results ('relevance', 'date', 'rating', 'viewCount', 'title')
            published_after (str): RFC 3339 formatted date-time (e.g., '2023-01-01T00:00:00Z')
            published_before (str): RFC 3339 formatted date-time
            video_duration (str): 'short' (<4min), 'medium' (4-20min), 'long' (>20min)
            video_definition (str): 'high' or 'standard'
            use_web_scraping (bool): Force use of web scraping instead of API
        
        Returns:
            Dict: Search results with video information
        """
        # Try API first if available and not forced to use web scraping
        if self.youtube and not use_web_scraping:
            try:
                return self._search_videos_api(query, max_results, order, published_after, 
                                             published_before, video_duration, video_definition)
            except Exception as e:
                logger.warning(f"API search failed: {e}")
                logger.info("Falling back to web scraping method...")
                print(self._fix_api_permissions())
        
        # Fallback to web scraping
        return self.search_videos_web_scraping(query, max_results)
    
    def _search_videos_api(self, query: str, max_results: int, order: str, 
                          published_after: Optional[str], published_before: Optional[str],
                          video_duration: Optional[str], video_definition: Optional[str]) -> Dict:
        """Original API-based search method"""
        try:
            # Ensure max_results doesn't exceed 50 (API limit per request)
            max_results = min(max_results, 50)
            
            search_params = {
                'part': 'id,snippet',
                'q': query,
                'type': 'video',
                'maxResults': max_results,
                'order': order,
                'regionCode': 'US',  # You can modify this as needed
                'relevanceLanguage': 'en'  # You can modify this as needed
            }
            
            # Add optional parameters if provided
            if published_after:
                search_params['publishedAfter'] = published_after
            if published_before:
                search_params['publishedBefore'] = published_before
            if video_duration:
                search_params['videoDuration'] = video_duration
            if video_definition:
                search_params['videoDefinition'] = video_definition
            
            # Execute the search
            search_response = self.youtube.search().list(**search_params).execute()
            
            # Process the results
            videos = []
            for item in search_response['items']:
                video_info = {
                    'video_id': item['id']['videoId'],
                    'title': item['snippet']['title'],
                    'description': item['snippet']['description'],
                    'channel_title': item['snippet']['channelTitle'],
                    'channel_id': item['snippet']['channelId'],
                    'published_at': item['snippet']['publishedAt'],
                    'thumbnail_url': item['snippet']['thumbnails']['default']['url'],
                    'video_url': f"https://www.youtube.com/watch?v={item['id']['videoId']}"
                }
                videos.append(video_info)
            
            result = {
                'total_results': search_response.get('pageInfo', {}).get('totalResults', 0),
                'results_per_page': search_response.get('pageInfo', {}).get('resultsPerPage', 0),
                'videos': videos,
                'query': query,
                'method': 'api'
            }
            
            logger.info(f"Successfully retrieved {len(videos)} videos for query: '{query}' using API")
            return result
            
        except HttpError as e:
            logger.error(f"HTTP error occurred: {e}")
            raise e
        except Exception as e:
            logger.error(f"API error occurred: {e}")
            raise e
    
    def search_by_tags(self, tags: List[str], **kwargs) -> Dict:
        """
        Search videos by tags
        
        Args:
            tags (List[str]): List of tags to search for
            **kwargs: Additional parameters for search_videos method
        
        Returns:
            Dict: Search results
        """
        query = ' '.join(tags)
        return self.search_videos(query, **kwargs)
    
    def search_by_title(self, title: str, **kwargs) -> Dict:
        """
        Search videos by title
        
        Args:
            title (str): Title to search for
            **kwargs: Additional parameters for search_videos method
        
        Returns:
            Dict: Search results
        """
        return self.search_videos(title, **kwargs)
    
    def search_by_topic(self, topic: str, **kwargs) -> Dict:
        """
        Search videos by topic
        
        Args:
            topic (str): Topic to search for
            **kwargs: Additional parameters for search_videos method
        
        Returns:
            Dict: Search results
        """
        return self.search_videos(topic, **kwargs)
    
    def get_video_links_only(self, query: str, max_results: int = 50, **kwargs) -> List[str]:
        """
        Get only the video links for a search query
        
        Args:
            query (str): Search query
            max_results (int): Maximum number of results
            **kwargs: Additional parameters for search_videos method
        
        Returns:
            List[str]: List of YouTube video URLs
        """
        result = self.search_videos(query, max_results, **kwargs)
        return [video['video_url'] for video in result.get('videos', [])]


def main():
    """
    Example usage of the YouTube Search Service with both API and web scraping
    """
    # Replace with your actual YouTube Data API v3 key (optional)
    API_KEY = "AIzaSyDzTHUS_7zPKz3Iw2B_cdXNMvCy7HzrS9A"  # Set to None to use only web scraping
    
    # Initialize the service
    if API_KEY == "AIzaSyDzTHUS_7zPKz3Iw2B_cdXNMvCy7HzrS9A":
        print("No API key provided. Using web scraping method only.")
        youtube_service = YouTubeSearchService()
    else:
        youtube_service = YouTubeSearchService(API_KEY)
    
    # Example 1: Search by topic using web scraping
    print("=== Searching by Topic: 'machine learning' (Web Scraping) ===")
    results = youtube_service.search_videos("machine learning", max_results=10, use_web_scraping=True)
    
    if 'error' not in results:
        print(f"Found {len(results['videos'])} videos using {results.get('method', 'unknown')} method")
        for i, video in enumerate(results['videos'][:5], 1):
            print(f"{i}. {video['title']}")
            print(f"   URL: {video['video_url']}")
            print(f"   Channel: {video['channel_title']}")
            print()
    else:
        print(f"Error: {results['error']}")
    
    # Example 2: Search by tags
    print("=== Searching by Tags ===")
    tags = ["python", "tutorial", "beginner"]
    results = youtube_service.search_by_tags(tags, max_results=5)
    
    if 'error' not in results:
        print(f"Found {len(results['videos'])} videos for tags: {tags}")
        for video in results['videos']:
            print(f"- {video['title']}: {video['video_url']}")
    else:
        print(f"Error: {results['error']}")
    
    # Example 3: Get only video links
    print("=== Getting Only Video Links ===")
    links = youtube_service.get_video_links_only("web development", max_results=5)
    print(f"Found {len(links)} video links:")
    for link in links:
        print(f"- {link}")
    
    # Example 4: Test different search methods
    print("\n=== Testing Web Scraping vs API ===")
    test_query = "react tutorial"
    
    # Force web scraping
    print("Using Web Scraping:")
    ws_results = youtube_service.search_videos(test_query, max_results=3, use_web_scraping=True)
    if 'error' not in ws_results:
        for video in ws_results['videos']:
            print(f"  - {video['title']}")
    
    # Try API (will fallback to web scraping if API fails)
    print("Using API (or fallback to web scraping):")
    api_results = youtube_service.search_videos(test_query, max_results=3, use_web_scraping=False)
    if 'error' not in api_results:
        for video in api_results['videos']:
            print(f"  - {video['title']}")




if __name__ == "__main__":
    main()