# crawl_lanet_session_save.py
import asyncio
from typing import List
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
import os
import re

# クロール対象URL（必要に応じて追加）
TARGET_URLS = [
    "https://lanet.sist.chukyo-u.ac.jp/",
    "https://lanet.sist.chukyo-u.ac.jp/activities",
    "https://lanet.sist.chukyo-u.ac.jp/societies",
    "https://lanet.sist.chukyo-u.ac.jp/researches",
    "https://lanet.sist.chukyo-u.ac.jp/jobs",
    "https://lanet.sist.chukyo-u.ac.jp/members",
    "https://lanet.sist.chukyo-u.ac.jp/links"
]

# 保存先ディレクトリ
OUTPUT_DIR = "lanet_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def sanitize_filename(url: str) -> str:
    return re.sub(r'[^\w\-_.]', '_', url.strip("/"))[:100]

async def crawl_sequential_and_save(urls: List[str]):
    print("\n=== Crawl4AI + セッション再利用 + Markdown保存 ===")

    browser_config = BrowserConfig(
        headless=True,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
    )

    crawl_config = CrawlerRunConfig(
        markdown_generator=DefaultMarkdownGenerator()
    )

    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()

    try:
        session_id = "lanet_session"
        for url in urls:
            result = await crawler.arun(
                url=url,
                config=crawl_config,
                session_id=session_id
            )

            if result.success:
                print(f"✅ Success: {url}")
                filename = sanitize_filename(url)
                path = os.path.join(OUTPUT_DIR, f"{filename}.txt")
                with open(path, "w", encoding="utf-8") as f:
                    f.write(result.markdown.raw_markdown or "")
                print(f"📄 保存: {path}")
            else:
                print(f"❌ Failed: {url} - {result.error_message}")
    finally:
        await crawler.close()
        print("✅ クロール完了（すべてのセッションを閉じました）")

async def main():
    await crawl_sequential_and_save(TARGET_URLS)

if __name__ == "__main__":
    asyncio.run(main())
