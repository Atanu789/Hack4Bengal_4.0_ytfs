import asyncio
import logging
import re
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from dotenv import load_dotenv
import requests
from pymongo import MongoClient
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)
from .embedding_service import EmbeddingService
from aiolimiter import AsyncLimiter
import os
from datetime import datetime
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from langchain_mongodb import MongoDBAtlasVectorSearch

logger = logging.getLogger(__name__)

load_dotenv()

MONGODB_ATLAS_CLUSTER_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("DB_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
ATLAS_VECTOR_SEARCH_INDEX_NAME = os.getenv("ATLAS_VECTOR_SEARCH_INDEX_NAME")
ATLAS_CONNECTION_STRING = MONGODB_ATLAS_CLUSTER_URI

class MongoDBEmbeddingSaver:
    def __init__(self):
        if not MONGODB_ATLAS_CLUSTER_URI:
            raise ValueError("MONGODB_URI environment variable is required")

        self.client = MongoClient(MONGODB_ATLAS_CLUSTER_URI)
        self.db = self.client[DB_NAME]
        self.collection = self.db[COLLECTION_NAME]

        # Initialize embedding service
        self.embedding_service = EmbeddingService()

        # Initialize vector store
        self.vector_store = MongoDBAtlasVectorSearch.from_connection_string(
            ATLAS_CONNECTION_STRING,
            namespace=f"{DB_NAME}.{COLLECTION_NAME}",
            embedding=self.embedding_service,
            index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
            relevance_score_fn="cosine",
        )

        print("MongoDB Vector Store initialized successfully!")

    async def save_video_embeddings(
        self, processed_video: "ProcessedVideo"
    ) -> Dict[str, any]:
        """Save optimized video embeddings to MongoDB (segments only)"""
        try:
            saved_documents = []

            # Save individual caption segments only
            if processed_video.caption_segments:
                segment_docs = []
                for i, segment in enumerate(processed_video.caption_segments):
                    segment_doc = {
                        "video_id": processed_video.video_info.video_id,
                     
                        "document_type": "caption_segment",
                        "segment_index": i,
                        "text_content": segment.text,
                        "start_time": segment.start_time,
                        "end_time": segment.end_time,
                        "duration": segment.duration,
                        "embedding": segment.embedding,
                        "is_merged": segment.is_merged,
                        "merged_count": segment.merged_count,
                        "created_at": datetime.now(),
                    }
                    segment_docs.append(segment_doc)

                # Bulk insert segments for efficiency
                if segment_docs:
                    segments_result = self.collection.insert_many(segment_docs)
                    for idx, inserted_id in enumerate(segments_result.inserted_ids):
                        saved_documents.append(
                            {
                                "type": "caption_segment",
                                "id": str(inserted_id),
                                "video_id": processed_video.video_info.video_id,
                                "segment_index": idx,
                            }
                        )

            return {
                "success": True,
                "saved_documents": saved_documents,
                "total_documents": len(saved_documents),
                "segments_saved": len(processed_video.caption_segments),
            }

        except Exception as e:
            logger.error(f"Error saving embeddings to MongoDB: {e}")
            return {
                "success": False,
                "error": str(e),
                "saved_documents": saved_documents,
            }

    async def search_similar_content(self, query_text: str, limit: int = 3) -> List[Dict]:
        """Search for similar content and return top segments with timestamps"""
        try:
            # Get embedding vector for the query text
            query_vector = self.embedding_service.embed_query(query_text)

            # Use $vectorSearch aggregation pipeline
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": ATLAS_VECTOR_SEARCH_INDEX_NAME,
                        "path": "embedding",
                        "queryVector": query_vector,
                        "numCandidates": 30,
                        "limit": limit
                    }
                },
                {
                    "$project": {
                        "_id": 0,
                        "content": "$text_content",
                        "video_id": "$video_id",
                        "title": "$title",
                        "author": "$author",
                        "start_time": "$start_time",
                        "end_time": "$end_time",
                        "duration": "$duration",
                        "is_merged": "$is_merged",
                        "merged_count": "$merged_count",
                        "similarity_score": {"$meta": "vectorSearchScore"}
                    }
                }
            ]

            results = list(self.collection.aggregate(pipeline))
            print(f"🔍 Found {results} similar segments for query '{query_text}'")
            # Format results with timestamps
            formatted_results = []
            for result in results:
                start_time = result.get("start_time", 0)
                end_time = result.get("end_time", 0)
                formatted_results.append({
                    "content": result.get("content", ""),
                    "video_id": result.get("video_id", ""),
                    "title": result.get("title", ""),
                    "author": result.get("author", ""),
                    "start_timestamp": self._seconds_to_timestamp(start_time),
                    "end_timestamp": self._seconds_to_timestamp(end_time),
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": result.get("duration", 0),
                    "is_merged": result.get("is_merged", False),
                    "merged_count": result.get("merged_count", 1),
                    "similarity_score": result.get("similarity_score", 0)
                })

            return formatted_results

        except Exception as e:
            logger.error(f"Error performing vector search: {e}")
            return []

    def _seconds_to_timestamp(self, seconds: float) -> str:
        """Convert seconds to readable timestamp (MM:SS or HH:MM:SS)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"

    def close(self):
        """Close MongoDB connection"""
        if hasattr(self, "client"):
            self.client.close()


@dataclass
class VideoInfo:
    """Simplified video information"""
    video_id: str
    title: str
    author: str


@dataclass
class CaptionSegment:
    """Represents a caption segment with timing and embedding"""
    text: str
    start_time: float
    end_time: float
    duration: float
    embedding: List[float]
    is_merged: bool = False
    merged_count: int = 1


@dataclass
class ProcessedVideo:
    video_info: VideoInfo
    caption_segments: List[CaptionSegment]


class RateLimitedEmbeddingService:
    """Wrapper around EmbeddingService with rate limiting and retry logic"""
    
    def __init__(self, requests_per_minute: int = 100):
        self.embedding_service = EmbeddingService()
        self.limiter = AsyncLimiter(requests_per_minute, 60)
        self.request_count = 0
        self.start_time = time.time()
        
    async def get_embedding_with_retry(self, text: str, max_retries: int = 3) -> List[float]:
        """Get embedding with rate limiting and retry logic"""
        for attempt in range(max_retries):
            try:
                async with self.limiter:
                    self.request_count += 1
                    if self.request_count % 10 == 0:
                        elapsed = time.time() - self.start_time
                        rate = self.request_count / (elapsed / 60)
                        print(f"📊 Embedding requests: {self.request_count}, Rate: {rate:.1f}/min")
                    
                    embedding = await asyncio.to_thread(
                        self.embedding_service.get_document_embedding, text
                    )
                    return embedding
                    
            except Exception as e:
                error_msg = str(e).lower()
                if "rate_limit_exceeded" in error_msg or "quota exceeded" in error_msg:
                    wait_time = (2 ** attempt) * 30
                    print(f"⚠️  Rate limit hit. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logger.warning(f"Failed to embed text '{text[:50]}...': {e}")
                    raise e
        
        raise Exception(f"Failed to get embedding after {max_retries} retries")


class YouTubeVideoService:
    def __init__(self, max_concurrent_embeddings: int = 2, requests_per_minute: int = 80, similarity_threshold: float = 0.85):
        self.session = requests.Session()
        self.limiter = AsyncLimiter(max_rate=2, time_period=1)
        self.embedding_service = RateLimitedEmbeddingService(requests_per_minute)
        self.semaphore = asyncio.Semaphore(max_concurrent_embeddings)
        self.mongo_saver = MongoDBEmbeddingSaver()
        self.similarity_threshold = similarity_threshold

    def extract_video_id(self, url: str) -> Optional[str]:
        patterns = [
            r"(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)",
            r"youtube\.com\/v\/([^&\n?#]+)",
            r"youtube\.com\/watch\?.*v=([^&\n?#]+)",
            r"(?:youtube\.com\/shorts\/)([^&\n?#]+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None

    async def get_basic_video_metadata(self, video_id: str) -> Dict:
        """Get only essential video metadata"""
        metadata = {}

        try:
            oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
            response = self.session.get(oembed_url, timeout=15)
            response.raise_for_status()

            data = response.json()
            metadata.update(
                {
                    "title": data.get("title", f"Video {video_id}"),
                    "author": data.get("author_name", "Unknown Author"),
                }
            )
        except Exception as e:
            logger.warning(f"oEmbed API failed: {e}")
            metadata = {
                "title": f"Video {video_id}",
                "author": "Unknown Author",
            }

        return metadata

    async def get_video_info(self, url: str) -> VideoInfo:
        """Get simplified video information"""
        try:
            video_id = self.extract_video_id(url)
            if not video_id:
                raise ValueError("Invalid YouTube URL")

            metadata = await self.get_basic_video_metadata(video_id)

            return VideoInfo(
                video_id=video_id,
                title=metadata["title"],
                author=metadata["author"],
            )

        except Exception as e:
            logger.error(f"Error getting video info: {e}")
            raise

    async def get_captions(self, url: str, languages: Optional[List[str]] = None) -> Dict[str, any]:
        """Get video captions"""
        try:
            video_id = self.extract_video_id(url)
            if not video_id:
                return {"success": False, "error": "Invalid YouTube URL"}

            try:
                transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            except (TranscriptsDisabled, VideoUnavailable):
                return {
                    "success": False,
                    "error": "Captions are disabled or video is unavailable for this video",
                }
            except NoTranscriptFound:
                return {"success": False, "error": "No captions found for this video"}

            if not languages:
                languages = ["en"]
            
            selected_transcript = None
            for lang in languages:
                try:
                    selected_transcript = transcript_list.find_transcript([lang])
                    break
                except NoTranscriptFound:
                    continue

            if not selected_transcript:
                available_transcripts = list(transcript_list)
                if available_transcripts:
                    selected_transcript = available_transcripts[0]

            if not selected_transcript:
                return {
                    "success": False,
                    "error": "No suitable captions found for the requested languages",
                }

            try:
                transcript_data = selected_transcript.fetch()
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Failed to fetch caption data: {str(e)}",
                }

            formatted_captions = []
            for item in transcript_data:
                caption_dict = {
                    "text": item.get("text", "") if isinstance(item, dict) else getattr(item, "text", ""),
                    "start": float(item.get("start", 0.0)) if isinstance(item, dict) else float(getattr(item, "start", 0.0)),
                    "duration": float(item.get("duration", 0.0)) if isinstance(item, dict) else float(getattr(item, "duration", 0.0)),
                }
                formatted_captions.append(caption_dict)

            return {
                "success": True,
                "video_id": video_id,
                "subtitles": formatted_captions,
            }

        except Exception as e:
            logger.error(f"Error getting captions: {e}")
            return {"success": False, "error": str(e)}

    def merge_similar_segments(self, segments_with_embeddings: List[Tuple[Dict, List[float]]]) -> List[CaptionSegment]:
        """Merge segments based on cosine similarity threshold"""
        if not segments_with_embeddings:
            return []

        # Extract embeddings for similarity calculation
        embeddings = np.array([emb for _, emb in segments_with_embeddings])
        segments = [seg for seg, _ in segments_with_embeddings]
        
        # Calculate cosine similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        merged_segments = []
        used_indices = set()
        
        print(f"🔄 Analyzing {len(segments)} segments for similarity merging (threshold: {self.similarity_threshold})")
        
        for i in range(len(segments)):
            if i in used_indices:
                continue
                
            # Find similar segments
            similar_indices = []
            for j in range(i, len(segments)):
                if j not in used_indices and similarity_matrix[i][j] >= self.similarity_threshold:
                    similar_indices.append(j)
            
            if len(similar_indices) > 1:
                # Merge similar segments
                merged_text = " ".join(segments[idx]["text"] for idx in similar_indices)
                start_time = min(segments[idx]["start"] for idx in similar_indices)
                end_times = [segments[idx]["start"] + segments[idx]["duration"] for idx in similar_indices]
                end_time = max(end_times)
                duration = end_time - start_time
                
                # Use the embedding of the first segment (representative)
                merged_embedding = segments_with_embeddings[i][1]
                
                merged_segment = CaptionSegment(
                    text=merged_text,
                    start_time=start_time,
                    end_time=end_time,
                    duration=duration,
                    embedding=merged_embedding,
                    is_merged=True,
                    merged_count=len(similar_indices)
                )
                
                merged_segments.append(merged_segment)
                used_indices.update(similar_indices)
                
                print(f"🔗 Merged {len(similar_indices)} segments (similarity: {similarity_matrix[i][similar_indices[1]]:.3f})")
                
            else:
                # Keep individual segment
                seg = segments[i]
                individual_segment = CaptionSegment(
                    text=seg["text"],
                    start_time=seg["start"],
                    end_time=seg["start"] + seg["duration"],
                    duration=seg["duration"],
                    embedding=segments_with_embeddings[i][1],
                    is_merged=False,
                    merged_count=1
                )
                
                merged_segments.append(individual_segment)
                used_indices.add(i)
        
        print(f"✅ Merged {len(segments)} segments into {len(merged_segments)} final segments")
        return merged_segments

    async def process_video_with_embeddings(
        self,
        url: str,
        languages: Optional[List[str]] = None,
        save_to_db: bool = True,
    ) -> Tuple[ProcessedVideo, Optional[Dict]]:
        """Process video with optimized embedding and similarity-based merging"""
        try:
            video_info = await self.get_video_info(url)
            
            # Get captions
            captions_result = await self.get_captions(url, languages=languages)
            
            if not captions_result["success"]:
                raise ValueError(f"Failed to get captions: {captions_result['error']}")

            subtitle_segments = captions_result.get("subtitles", [])
            
            if not subtitle_segments:
                raise ValueError("No subtitle segments found")

            print(f"🧠 Generating embeddings for {len(subtitle_segments)} segments...")
            
            # Generate embeddings for all segments
            segments_with_embeddings = []
            
            async def process_segment(segment):
                text = segment.get("text", "").strip()
                if not text:
                    return None
                try:
                    async with self.semaphore:
                        embedding = await self.embedding_service.get_embedding_with_retry(text)
                    return (segment, embedding)
                except Exception as e:
                    logger.warning(f"Failed to embed segment '{text[:50]}...': {e}")
                    return None

            # Process segments with rate limiting
            results = await asyncio.gather(
                *[process_segment(seg) for seg in subtitle_segments]
            )
            
            segments_with_embeddings = [r for r in results if r is not None]
            
            # Merge similar segments using cosine similarity
            print(f"🔍 Applying cosine similarity merging with threshold {self.similarity_threshold}")
            caption_segments = self.merge_similar_segments(segments_with_embeddings)

            processed_video = ProcessedVideo(
                video_info=video_info,
                caption_segments=caption_segments,
            )

            # Save to MongoDB if requested
            save_result = None
            if save_to_db:
                print("💾 Saving optimized embeddings to MongoDB...")
                save_result = await self.mongo_saver.save_video_embeddings(processed_video)
                
                if save_result["success"]:
                    print(f"✅ Successfully saved {save_result['total_documents']} segments to MongoDB")
                else:
                    print(f"❌ Failed to save to MongoDB: {save_result.get('error')}")

            return processed_video, save_result

        except Exception as e:
            logger.error(f"Error processing video with embeddings: {e}")
            raise

    async def search_videos(self, query: str, limit: int = 3) -> List[Dict]:
        """Search for similar videos and return top segments with timestamps"""
        return await self.mongo_saver.search_similar_content(query, limit)

    def display_segments_summary(self, processed_video: ProcessedVideo, max_segments: int = 5):
        """Display summary of processed segments"""
        print(f"\n📊 PROCESSED SEGMENTS SUMMARY")
        print("=" * 60)
        print(f"Total segments: {len(processed_video.caption_segments)}")
        
        merged_count = sum(1 for seg in processed_video.caption_segments if seg.is_merged)
        individual_count = len(processed_video.caption_segments) - merged_count
        
        print(f"Merged segments: {merged_count}")
        print(f"Individual segments: {individual_count}")
        
        print(f"\n📝 SAMPLE SEGMENTS:")
        print("-" * 60)
        
        segments_to_show = min(max_segments, len(processed_video.caption_segments))
        
        for i, segment in enumerate(processed_video.caption_segments[:segments_to_show]):
            status = "MERGED" if segment.is_merged else "INDIVIDUAL"
            print(f"\n{i+1}. [{status}] {self._seconds_to_timestamp(segment.start_time)} - {self._seconds_to_timestamp(segment.end_time)}")
            print(f"   Text: {segment.text[:100]}{'...' if len(segment.text) > 100 else ''}")
            if segment.is_merged:
                print(f"   Merged from {segment.merged_count} segments")

    def _seconds_to_timestamp(self, seconds: float) -> str:
        """Convert seconds to readable timestamp"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"

    def close(self):
        """Close all connections"""
        if hasattr(self, "mongo_saver"):
            self.mongo_saver.close()
        if hasattr(self, "session"):
            self.session.close()


async def main():
    """Main function to test the optimized YouTube video processing"""
    test_urls = ["https://www.youtube.com/watch?v=s2skans2dP4"]

    # Initialize service with similarity threshold
    service = YouTubeVideoService(similarity_threshold=0.85)

    try:
        for url in test_urls:
            print(f"\n{'=' * 60}")
            print(f"Processing URL: {url}")
            print("=" * 60)

            try:
                # Process video with embeddings
                processed_video, save_result = await service.process_video_with_embeddings(
                    url, languages=["en"], save_to_db=True
                )

                print(f"\n📹 Video Info:")
                print(f"   Title: {processed_video.video_info.title}")
                print(f"   Author: {processed_video.video_info.author}")

                # Display segments summary
                service.display_segments_summary(processed_video, max_segments=5)

                # Show MongoDB save results
                if save_result:
                    print(f"\n💾 MongoDB Save Results:")
                    print(f"   Success: {save_result['success']}")
                    print(f"   Segments saved: {save_result.get('segments_saved', 0)}")

                # Test search functionality
                print(f"\n🔍 SEARCH TEST - Top 3 Similar Segments:")
                print("=" * 50)
                search_query = "React Router"
                search_results = await service.search_videos(search_query, limit=3)
                
                if search_results:
                    print(f"Query: '{search_query}'")
                    print(f"Found {len(search_results)} similar segments:\n")
                    
                    for i, result in enumerate(search_results, 1):
                        print(f"{i}. [{result['start_timestamp']} - {result['end_timestamp']}] Score: {result['similarity_score']:.4f}")
                        print(f"   Video: {result['title']} by {result['author']}")
                        print(f"   Content: {result['content'][:150]}{'...' if len(result['content']) > 150 else ''}")
                        if result['is_merged']:
                            print(f"   [Merged from {result['merged_count']} segments]")
                        print()
                else:
                    print("No search results found.")

                print(f"✅ Successfully processed and saved video with optimized embeddings!")

            except Exception as e:
                print(f"❌ Error processing {url}: {e}")
                logger.exception("Full error traceback:")

    finally:
        service.close()

    print(f"\n{'=' * 60}")
    print("Processing complete!")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run the main function
    asyncio.run(main())