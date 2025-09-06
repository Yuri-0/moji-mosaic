"""BERT-based keyword extraction using TF-IDF and optional vLLM acceleration."""

import logging
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer
from vllm import LLM


class KeywordExtractor:
    """Extract keywords from text using BERT models via vLLM."""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_keywords: int = 5,
        use_vllm: bool = True,
        gpu_memory_utilization: float = 0.3,
        cache_dir: Optional[Path] = None,
    ):
        """Initialize the keyword extractor.
        
        Args:
            model_name: HuggingFace model name for BERT-based extraction
            max_keywords: Maximum number of keywords to extract
            use_vllm: Whether to use vLLM for acceleration
            gpu_memory_utilization: GPU memory utilization for vLLM
            cache_dir: Directory to cache models
        """
        self.model_name = model_name
        self.max_keywords = max_keywords
        self.use_vllm = use_vllm
        self.cache_dir = cache_dir or Path.home() / ".moji_mosaic" / "models"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._llm = None
        self._tokenizer = None
        self._tfidf_vectorizer = None
        
        # Initialize models
        self._initialize_models(gpu_memory_utilization)
    
    def _initialize_models(self, gpu_memory_utilization: float) -> None:
        """Initialize the BERT model and tokenizer."""
        try:
            if self.use_vllm and LLM is not None:
                # Initialize vLLM for efficient inference
                self._llm = LLM(
                    model=self.model_name,
                    gpu_memory_utilization=gpu_memory_utilization,
                    download_dir=str(self.cache_dir),
                    trust_remote_code=True,
                )
                logging.info(f"Initialized vLLM with model: {self.model_name}")
            
            if AutoTokenizer is not None:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    cache_dir=str(self.cache_dir)
                )
                logging.info(f"Initialized tokenizer for: {self.model_name}")
            
        except Exception as e:
            logging.warning(f"Failed to initialize vLLM/transformers: {e}")
            self.use_vllm = False
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for keyword extraction."""
        # Remove special characters and normalize whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip().lower()
    
    def _similarity_ratio(self, word1: str, word2: str) -> float:
        """Calculate similarity ratio between two words."""
        return SequenceMatcher(None, word1.lower(), word2.lower()).ratio()
    
    def _filter_similar_keywords(self, keywords: List[str], similarity_threshold: float = 0.7) -> List[str]:
        """Filter out similar keywords to ensure diversity."""
        if not keywords:
            return keywords
        
        # Pre-process keywords to remove obvious duplicates and clean them
        cleaned_keywords = []
        seen_words = set()
        
        for keyword in keywords:
            # Clean the keyword
            clean_kw = keyword.strip().lower()
            
            # Skip if empty or too short
            if len(clean_kw) < 2:
                continue
                
            # Skip if it's a repeated word (like "world world")
            words_in_kw = clean_kw.split()
            if len(words_in_kw) > 1 and len(set(words_in_kw)) == 1:
                # This is a repeated word, use just the single word
                clean_kw = words_in_kw[0]
            
            # Skip if we've already seen this exact word
            if clean_kw in seen_words:
                continue
                
            seen_words.add(clean_kw)
            cleaned_keywords.append(keyword)  # Keep original case for display
        
        if not cleaned_keywords:
            return []
        
        filtered = [cleaned_keywords[0]]  # Always keep the first (highest scored) keyword
        
        for keyword in cleaned_keywords[1:]:
            # Check if this keyword is too similar to any already selected keyword
            is_similar = False
            for selected in filtered:
                # Check for exact substring matches
                if keyword.lower() in selected.lower() or selected.lower() in keyword.lower():
                    is_similar = True
                    break
                # Check for high similarity ratio
                if self._similarity_ratio(keyword, selected) > similarity_threshold:
                    is_similar = True
                    break
            
            if not is_similar:
                filtered.append(keyword)
                
            # Stop if we have enough diverse keywords
            if len(filtered) >= self.max_keywords:
                break
        
        return filtered
    
    def _extract_with_tfidf(self, text: str) -> List[str]:
        """Improved keyword extraction using TF-IDF with diversity filtering."""
        # Split text into sentences for TF-IDF analysis
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if len(sentences) < 2:
            sentences = [text]  # Use whole text if too few sentences
        
        try:
            # Create a new vectorizer each time to avoid issues with document count
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                max_df=1.0  # Set to 1.0 to avoid max_df < min_df issues
            )
            
            tfidf_matrix = vectorizer.fit_transform(sentences)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get average TF-IDF scores across all sentences
            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            
            # Get more candidates than needed for diversity filtering
            num_candidates = min(len(feature_names), self.max_keywords * 3)
            top_indices = mean_scores.argsort()[-num_candidates:][::-1]
            candidate_keywords = [feature_names[i] for i in top_indices if mean_scores[i] > 0]
            
            # Filter out similar keywords to ensure diversity
            diverse_keywords = self._filter_similar_keywords(candidate_keywords)
            
            return diverse_keywords[:self.max_keywords]
            
        except Exception as e:
            logging.warning(f"TF-IDF extraction failed: {e}")
            # Ultimate fallback: return most frequent words with diversity filtering
            words = text.split()
            word_freq = {}
            for word in words:
                if len(word) > 3:  # Filter short words
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Get candidate words sorted by frequency and apply diversity filtering
            candidate_words = sorted(word_freq.keys(), key=word_freq.get, reverse=True)
            diverse_words = self._filter_similar_keywords(candidate_words)
            
            return diverse_words[:self.max_keywords]
    
    def _extract_with_bert_prompt(self, text: str) -> List[str]:
        """Extract keywords using BERT model with prompting.
        
        Note: Currently falls back to TF-IDF due to model compatibility issues.
        """
        logging.info("Using TF-IDF extraction instead of vLLM due to model compatibility")
        return self._extract_with_tfidf(text)
    
    def extract_keywords(
        self, 
        text: str, 
        method: str = "bert",
        context: Optional[str] = None
    ) -> List[str]:
        """Extract keywords from the input text.
        
        Args:
            text: Input text to extract keywords from
            method: Extraction method ("bert", "tfidf", "hybrid")
            context: Additional context to guide keyword extraction
            
        Returns:
            List of extracted keywords
        """
        if not text.strip():
            return []
        
        # Preprocess text
        processed_text = self._preprocess_text(text)
        if context:
            processed_text = f"{self._preprocess_text(context)} {processed_text}"
        
        # Extract keywords based on method
        if method == "bert" and self.use_vllm:
            keywords = self._extract_with_bert_prompt(processed_text)
        elif method == "tfidf":
            keywords = self._extract_with_tfidf(processed_text)
        elif method == "hybrid":
            # Combine both methods
            bert_keywords = self._extract_with_bert_prompt(processed_text) if self.use_vllm else []
            tfidf_keywords = self._extract_with_tfidf(processed_text)
            
            # Merge and deduplicate
            all_keywords = bert_keywords + tfidf_keywords
            seen = set()
            keywords = []
            for kw in all_keywords:
                if kw.lower() not in seen:
                    keywords.append(kw)
                    seen.add(kw.lower())
                if len(keywords) >= self.max_keywords:
                    break
        else:
            keywords = self._extract_with_tfidf(processed_text)
        
        # Ensure we don't exceed max_keywords
        return keywords[:self.max_keywords]
    
    def get_keyword_scores(self, text: str, keywords: List[str]) -> Dict[str, float]:
        """Get relevance scores for keywords in the context of the text."""
        if not keywords or not text.strip():
            return {}
        
        try:
            # Create a new vectorizer for scoring to avoid issues
            vectorizer = TfidfVectorizer(
                vocabulary=keywords,
                stop_words='english'
            )
            
            # Fit on the text and get scores
            tfidf_matrix = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            
            return dict(zip(feature_names, scores))
            
        except Exception as e:
            logging.warning(f"Keyword scoring failed: {e}")
            # Return equal scores as fallback
            return {kw: 1.0 for kw in keywords}
    
    def analyze_text_themes(self, text: str) -> Dict[str, Any]:
        """Analyze the text and return thematic information."""
        keywords = self.extract_keywords(text, method="hybrid")
        keyword_scores = self.get_keyword_scores(text, keywords)
        
        return {
            'keywords': keywords,
            'keyword_scores': keyword_scores,
            'text_length': len(text),
            'word_count': len(text.split()),
            'top_keyword': keywords[0] if keywords else None,
            'theme_strength': sum(keyword_scores.values()) / len(keyword_scores) if keyword_scores else 0.0,
        }
