#!/usr/bin/env python3
"""
Web scraper to collect training data for AI text detection.
Scrapes various websites to get human-written content samples.
"""

import requests
import time
import random
import json
import os
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re
from datetime import datetime
import feedparser
import yaml

class WebScraper:
    def __init__(self, delay_range: tuple = (1, 3)):
        """Initialize the web scraper."""
        self.delay_range = delay_range
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Create output directory
        os.makedirs('scraped_data', exist_ok=True)
        os.makedirs('scraped_data/human', exist_ok=True)
        os.makedirs('scraped_data/ai', exist_ok=True)
    
    def scrape_website(self, url: str, max_pages: int = 5, min_words: int = 400) -> List[Dict]:
        """Scrape a single website for content.

        Strategy:
        - Load the landing page
        - Extract likely article links
        - Visit up to max_pages article pages
        - Extract main content using robust heuristics
        - Keep only articles with at least min_words words
        """
        
        print(f" Scraping: {url}")
        
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            base_url = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
            article_links = self.extract_article_links(soup, base_url)
            if not article_links:
                print("  No article links found on landing page; attempting to parse main content directly.")
                text = self.extract_main_text(soup)
                if len(text.split()) >= min_words:
                    return [{
                        'url': url,
                        'title': self.extract_page_title(soup) or 'Untitled',
                        'text': text,
                        'word_count': len(text.split()),
                        'timestamp': datetime.now().isoformat()
                    }]
                return []
            
            # Visit individual article links
            articles: List[Dict] = []
            seen_urls = set()
            for link in article_links[:max_pages]:
                if link in seen_urls:
                    continue
                seen_urls.add(link)
                try:
                    time.sleep(random.uniform(0.5, 1.2))
                    article_resp = self.session.get(link, timeout=15)
                    article_resp.raise_for_status()
                    article_soup = BeautifulSoup(article_resp.content, 'html.parser')
                    text = self.extract_main_text(article_soup)
                    if len(text.split()) < min_words:
                        continue
                    title = self.extract_page_title(article_soup) or 'Untitled'
                    articles.append({
                        'url': link,
                        'title': title,
                        'text': text,
                        'word_count': len(text.split()),
                        'timestamp': datetime.now().isoformat()
                    })
                except Exception as e:
                    print(f"    Skipping link due to error: {link} ({e})")
                    continue
            
            print(f"   Collected {len(articles)} full articles")
            return articles
        
        except Exception as e:
            print(f"   Error scraping {url}: {e}")
            return []
    
    def extract_clean_text(self, element) -> str:
        """Extract clean text from HTML element."""
        
        # Remove script and style elements
        for script in element(["script", "style", "nav", "header", "footer", "aside"]):
            script.decompose()
        
        # Get text
        text = element.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def extract_main_text(self, soup: BeautifulSoup) -> str:
        """Extract the main article text using robust heuristics.

        Approach:
        - Prefer known article containers
        - Fallback to concatenating paragraph tags from the main content area
        - Avoid nav/footers/sidebars and non-article elements
        """
        # Candidate selectors commonly used across publishers
        candidate_selectors = [
            'article',
            'div[itemprop="articleBody"]',
            'div[id*="article"]',
            'div[class*="article-body"]',
            'div[class*="article__content"]',
            'div[class*="story-content"]',
            'div[class*="StoryBodyCompanionColumn"]',
            'div[class*="post-content"]',
            'div.mw-parser-output',
            'div#content',
            '#storytext',
            'div[class*="storytext"]',
            'main',
            'section[name="articleBody"]',
            'div[data-testid="story-text"]',
            'div[data-testid="ArticleBody"]',
            'section[role="main"] article'
        ]
        
        # Try candidates and gather paragraphs
        paragraphs: List[str] = []
        for selector in candidate_selectors:
            container = soup.select_one(selector)
            if container:
                # Prefer p tags to avoid unrelated text
                p_tags = container.find_all('p')
                if p_tags:
                    paragraphs = [self.extract_clean_text(p) for p in p_tags if self.extract_clean_text(p)]
                    break
        
        # BBC and some others use repeated text-block components
        if not paragraphs:
            text_blocks = soup.select('div[data-component="text-block"]')
            if text_blocks:
                paragraphs = [self.extract_clean_text(tb) for tb in text_blocks if self.extract_clean_text(tb)]
        
        # Fallback: collect all <p> under body, excluding common noisy sections
        if not paragraphs:
            body = soup.find('body')
            if body:
                for tag in body.find_all(['nav', 'header', 'footer', 'aside']):
                    tag.decompose()
                paragraphs = [
                    self.extract_clean_text(p) for p in body.find_all('p')
                    if self.extract_clean_text(p)
                ]
        
        # Join paragraphs into text
        text = '\n\n'.join(paragraphs)
        text = re.sub(r'\n{3,}', '\n\n', text).strip()
        return text
    
    def extract_page_title(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract title from the full page soup."""
        if soup.title and soup.title.get_text(strip=True):
            return soup.title.get_text(strip=True)
        h1 = soup.find('h1')
        if h1 and h1.get_text(strip=True):
            return h1.get_text(strip=True)
        return None
    
    def extract_title(self, element) -> str:
        """Extract title from element (backwards compatibility)."""
        title = element.find('title')
        if title:
            return title.get_text().strip()
        h1 = element.find('h1')
        if h1:
            return h1.get_text().strip()
        return "Untitled"

    def normalize_url(self, href: str, base_url: str) -> Optional[str]:
        """Normalize and filter URLs to likely article links on the same domain."""
        if not href:
            return None
        href = href.strip()
        if href.startswith('#'):
            return None
        # Join relative URLs
        full_url = urljoin(base_url, href)
        parsed_base = urlparse(base_url)
        parsed_full = urlparse(full_url)
        # Keep on same root domain when possible (allow subdomains)
        base_host = parsed_base.netloc.split(':')[0]
        full_host = parsed_full.netloc.split(':')[0]
        if not (full_host == base_host or full_host.endswith('.' + base_host)):
            return None
        path = parsed_full.path.lower()
        # Heuristics to filter non-article pages
        exclude_substrings = [
            '/tag/', '/tags/', '/topic/', '/topics/', '/video', '/videos', '/live', '/gallery', '/photos', '/about', '/contact', '/privacy', '/terms'
        ]
        if any(sub in path for sub in exclude_substrings):
            return None
        # Likely article if path is deep enough or matches date/story patterns
        date_pattern = re.compile(r"/20\d{2}/(0?[1-9]|1[0-2])/(0?[1-9]|[12]\d|3[01])/")
        keywords = ('/story/', '/article', '/news/', '/202', '/opinions/', '/science/', '/technology/', '/world/', '/business/')
        deep_enough = path.count('/') >= 2
        looks_like_article = deep_enough or bool(date_pattern.search(path)) or any(k in path for k in keywords)
        if not looks_like_article:
            return None
        return full_url

    def extract_article_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract a list of likely article links from a landing page soup."""
        links: List[str] = []
        seen = set()
        for a in soup.find_all('a', href=True):
            normalized = self.normalize_url(a.get('href'), base_url)
            if normalized and normalized not in seen:
                seen.add(normalized)
                links.append(normalized)
        # Keep the first N links to avoid over-scraping
        return links[:50]
    
    def save_articles(self, articles: List[Dict], source: str):
        """Save scraped articles to files."""
        
        for i, article in enumerate(articles):
            # Create filename
            safe_title = re.sub(r'[^\w\s-]', '', article['title'])
            safe_title = re.sub(r'[-\s]+', '-', safe_title)
            filename = f"scraped_data/human/{source}_{i+1}_{safe_title[:50]}.txt"
            
            # Save article
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"Source: {article['url']}\n")
                f.write(f"Title: {article['title']}\n")
                f.write(f"Word Count: {article['word_count']}\n")
                f.write(f"Timestamp: {article['timestamp']}\n")
                f.write("-" * 80 + "\n\n")
                f.write(article['text'])
            
            print(f"     Saved: {filename}")
    
    def scrape_multiple_sites(self, sites: List[Dict], min_words: int = 400) -> Dict[str, List[Dict]]:
        """Scrape multiple websites."""
        
        all_results = {}
        
        for site in sites:
            print(f"\n Scraping {site['name']}...")
            
            # If RSS feed provided, use it for reliable article URLs
            rss_url = site.get('rss')
            articles: List[Dict] = []
            if rss_url:
                try:
                    feed = feedparser.parse(rss_url)
                    base_url = f"{urlparse(site['url']).scheme}://{urlparse(site['url']).netloc}"
                    count = 0
                    for entry in feed.entries:
                        if count >= site.get('max_pages', 5):
                            break
                        link = entry.get('link')
                        link = self.normalize_url(link, base_url) or link
                        if not link:
                            continue
                        try:
                            time.sleep(random.uniform(0.5, 1.0))
                            resp = self.session.get(link, timeout=15)
                            resp.raise_for_status()
                            soup = BeautifulSoup(resp.content, 'html.parser')
                            text = self.extract_main_text(soup)
                            if len(text.split()) < min_words:
                                continue
                            title = self.extract_page_title(soup) or entry.get('title') or 'Untitled'
                            articles.append({
                                'url': link,
                                'title': title,
                                'text': text,
                                'word_count': len(text.split()),
                                'timestamp': datetime.now().isoformat()
                            })
                            count += 1
                        except Exception as e:
                            print(f"    RSS link failed: {link} ({e})")
                except Exception as e:
                    print(f"    RSS parsing failed for {rss_url}: {e}")
            else:
                articles = self.scrape_website(site['url'], site.get('max_pages', 5), min_words=min_words)
            
            if articles:
                all_results[site['name']] = articles
                self.save_articles(articles, site['name'])
            
            # Random delay between requests
            delay = random.uniform(*self.delay_range)
            print(f"  Waiting {delay:.1f}s...")
            time.sleep(delay)
        
        return all_results
    
    def generate_summary_report(self, results: Dict[str, List[Dict]]):
        """Generate a summary report of scraped data."""
        
        total_articles = sum(len(articles) for articles in results.values())
        total_words = sum(
            sum(article['word_count'] for article in articles)
            for articles in results.values()
        )
        
        report = {
            'scraping_date': datetime.now().isoformat(),
            'total_sites': len(results),
            'total_articles': total_articles,
            'total_words': total_words,
            'site_breakdown': {
                site: {
                    'articles': len(articles),
                    'words': sum(article['word_count'] for article in articles)
                }
                for site, articles in results.items()
            }
        }
        
        # Save report
        with open('scraped_data/scraping_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n Scraping Summary:")
        print(f"  Sites scraped: {len(results)}")
        print(f"  Total articles: {total_articles}")
        print(f"  Total words: {total_words:,}")
        print(f"  Report saved: scraped_data/scraping_report.json")

def get_target_sites() -> List[Dict]:
    """Get list of target websites to scrape."""
    
    return [
        # News sites (likely human-written)
        {
            'name': 'reuters',
            'url': 'https://www.reuters.com',
            'rss': 'http://feeds.reuters.com/reuters/topNews',
            'max_pages': 3,
            'description': 'International news agency'
        },
        {
            'name': 'bbc_news',
            'url': 'https://www.bbc.com/news',
            'rss': 'https://feeds.bbci.co.uk/news/rss.xml',
            'max_pages': 3,
            'description': 'BBC News'
        },
        {
            'name': 'npr',
            'url': 'https://www.npr.org',
            'rss': 'https://feeds.npr.org/1001/rss.xml',
            'max_pages': 3,
            'description': 'National Public Radio'
        },
        
        # Academic/Educational sites
        {
            'name': 'wikipedia',
            'url': 'https://en.wikipedia.org/wiki/Special:Random',
            'max_pages': 5,
            'description': 'Wikipedia articles'
        },
        {
            'name': 'stanford_news',
            'url': 'https://news.stanford.edu',
            'max_pages': 3,
            'description': 'Stanford University news'
        },
        
        # Tech blogs (mix of human/AI)
        {
            'name': 'techcrunch',
            'url': 'https://techcrunch.com',
            'rss': 'https://techcrunch.com/feed/',
            'max_pages': 3,
            'description': 'Technology news and analysis'
        },
        {
            'name': 'arstechnica',
            'url': 'https://arstechnica.com',
            'rss': 'https://feeds.arstechnica.com/arstechnica/index',
            'max_pages': 3,
            'description': 'Technology and science news'
        },
        
        # Personal blogs (likely human)
        {
            'name': 'medium',
            'url': 'https://medium.com',
            'max_pages': 3,
            'description': 'Personal and professional articles'
        },
        
        # Government/Institutional sites
        {
            'name': 'nasa',
            'url': 'https://www.nasa.gov/news',
            'max_pages': 3,
            'description': 'NASA news and updates'
        },
        
        # Science/Research sites
        {
            'name': 'nature',
            'url': 'https://www.nature.com/news',
            'max_pages': 3,
            'description': 'Scientific research news'
        },
        
        # Local news (likely human)
        {
            'name': 'local_news',
            'url': 'https://www.sfchronicle.com',
            'max_pages': 3,
            'description': 'Local San Francisco news'
        },
        
        # Lifestyle and Food blogs
        {
            'name': 'food_network',
            'url': 'https://www.foodnetwork.com/recipes',
            'max_pages': 3,
            'description': 'Food recipes and cooking tips'
        },
        {
            'name': 'allrecipes',
            'url': 'https://www.allrecipes.com',
            'rss': 'https://www.allrecipes.com/feed/',
            'max_pages': 3,
            'description': 'Community-driven recipe sharing'
        },
        {
            'name': 'mindbodygreen',
            'url': 'https://www.mindbodygreen.com',
            'max_pages': 3,
            'description': 'Wellness and lifestyle content'
        },
        
        # Philosophy and Life blogs
        {
            'name': 'brainpickings',
            'url': 'https://www.themarginalian.org',
            'max_pages': 3,
            'description': 'Philosophy, creativity, and life insights'
        },
        {
            'name': 'zenhabits',
            'url': 'https://zenhabits.net',
            'max_pages': 3,
            'description': 'Minimalism and mindfulness'
        },
        
        # Business and Professional
        {
            'name': 'hbr',
            'url': 'https://hbr.org',
            'max_pages': 3,
            'description': 'Harvard Business Review'
        },
        {
            'name': 'forbes',
            'url': 'https://www.forbes.com',
            'max_pages': 3,
            'description': 'Business and entrepreneurship'
        },
        
        # Personal Development
        {
            'name': 'tim_ferriss',
            'url': 'https://tim.blog',
            'max_pages': 3,
            'description': 'Personal development and productivity'
        },
        {
            'name': 'seth_godin',
            'url': 'https://seths.blog',
            'max_pages': 3,
            'description': 'Marketing and life philosophy'
        }
    ]

def load_config(config_path: str = "config.yaml") -> Dict:
    """Load YAML configuration if available."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}

def get_sites_from_config(cfg: Dict) -> List[Dict]:
    """Build sites list from config, applying defaults and limits.

    - Uses training.sites if present. Each site must have name and url.
    - Fills missing max_pages from training.human_articles_per_site.
    - Applies training.max_sites limit.
    """
    training_cfg = (cfg or {}).get('training', {}) or {}
    raw_sites = training_cfg.get('sites') or []
    if not raw_sites:
        return []

    default_max_pages = int(training_cfg.get('human_articles_per_site', 5))
    max_sites = training_cfg.get('max_sites')
    processed: List[Dict] = []
    for site in raw_sites:
        try:
            name = site.get('name')
            url = site.get('url')
            if not name or not url:
                continue
            item = {
                'name': name,
                'url': url,
            }
            if site.get('rss'):
                item['rss'] = site['rss']
            item['max_pages'] = int(site.get('max_pages', default_max_pages))
            if site.get('description'):
                item['description'] = site['description']
            processed.append(item)
        except Exception:
            continue

    if isinstance(max_sites, int) and max_sites > 0:
        processed = processed[:max_sites]

    return processed

def main():
    """Main scraping function."""
    
    print(" Web Scraper for AI Text Detection Training")
    print("=" * 60)
    
    # Load config and resolve sites and parameters
    cfg = load_config()
    training_cfg = (cfg or {}).get('training', {}) or {}
    min_words = int(training_cfg.get('min_human_words', 400))
    sites = get_sites_from_config(cfg) or get_target_sites()
    print(f" Target sites: {len(sites)}")
    
    # Create scraper
    scraper = WebScraper(delay_range=(2, 5))  # Be respectful with delays
    
    # Start scraping
    print("\n Starting web scraping...")
    results = scraper.scrape_multiple_sites(sites, min_words=min_words)
    
    # Generate report
    scraper.generate_summary_report(results)
    
    print(f"\n Scraping completed!")
    print(f" Check the 'scraped_data/human/' directory for collected articles.")

if __name__ == "__main__":
    main()
