"""Freeze Wikipedia for 180 selected FActScore subjects, or finalize offline; no inference."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from urllib.parse import parse_qs, urlparse

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from factscore_selection import (
    ARCHIVE_SHA256, FACTSCORE_PATH, FACTSCORE_MEMBER, PROMPT_TEMPLATE, SELECTED_COUNT,
    SOURCE_RECEIPT_PATH, SOURCE_RECEIPT_SHA256, select_archive,
)

ROOT = Path(__file__).resolve().parents[1]
API = 'https://en.wikipedia.org/w/api.php'
EXTRACTOR = 'wikipedia-oldid-html-text-v1'
UI_SELECTORS = 'script, style, .mw-editsection, .navbox, .vertical-navbox, .toc, .portalbox, .hatnote, .ambox, .ombox, .tmbox, .fmbox, .sistersitebox, .mw-indicators, .noprint'
EXTRACTION_RULES = f'BeautifulSoup html.parser; remove {UI_SELECTORS}; retain other text including tables/citations; strip blank lines.'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8')


def immutable(path, data):
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f'Refusing to change frozen artifact: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        # Publish a complete file without overwriting any competing destination.
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise ValueError(f'Refusing to change frozen artifact: {path}')
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def subject_selection(root):
    """Read only the anchored receipt/archive and use the shared exact selection."""
    receipt_bytes = (root / SOURCE_RECEIPT_PATH).read_bytes()
    if digest(receipt_bytes) != SOURCE_RECEIPT_SHA256:
        raise ValueError('FActScore source receipt hash changed')
    receipt = json.loads(receipt_bytes)
    entries = [r for r in receipt['files'] if r['path'] == FACTSCORE_PATH]
    if len(entries) != 1 or entries[0]['sha256'] != ARCHIVE_SHA256:
        raise ValueError('FActScore source archive receipt changed')
    data = (root / FACTSCORE_PATH).read_bytes()
    if len(data) != entries[0]['bytes']:
        raise ValueError('FActScore source archive hash changed')
    return select_archive(data)


def topics(root):
    return [topic for _, topic in subject_selection(root)[0]]


def clean_html(html):
    soup = BeautifulSoup(html, 'html.parser')
    # Retain paragraphs, tables and citation text; discard UI/script content only.
    for element in soup.select(UI_SELECTORS):
        element.decompose()
    return '\n'.join(line.strip() for line in soup.get_text('\n').splitlines() if line.strip()) + '\n'


def resolved_page(query_record, topic):
    url = urlparse(query_record['url'])
    params = parse_qs(url.query)
    if (url.scheme != 'https' or url.netloc != 'en.wikipedia.org' or url.path != '/w/api.php'
            or params.get('titles') != [topic] or params.get('action') != ['query']):
        raise ValueError('Query URL does not identify the requested Wikipedia topic')
    pages = query_record['response'].get('query', {}).get('pages', [])
    if len(pages) != 1 or 'missing' in pages[0] or 'invalid' in pages[0]:
        raise ValueError('Topic did not resolve to exactly one existing page')
    page = pages[0]
    resolved_title = topic
    query = query_record['response']['query']
    for group in ['normalized', 'converted', 'redirects']:
        links = {item['from']: item['to'] for item in query.get(group, [])}
        visited = set()
        while resolved_title in links:
            if resolved_title in visited:
                raise ValueError('Cyclic Wikipedia title resolution')
            visited.add(resolved_title)
            resolved_title = links[resolved_title]
    if page.get('title') != resolved_title:
        raise ValueError('Returned title is not the requested topic or its recorded redirect')
    if 'disambiguation' in page.get('pageprops', {}):
        raise ValueError('Topic resolves to a disambiguation page; no automatic entity substitution')
    if len(page.get('revisions', [])) != 1 or page.get('ns') != 0:
        raise ValueError('Expected one revision of an article page')
    return page


def fetch_one(root, topic):
    key = digest(topic.encode('utf-8'))[:20]
    directory = root / 'data/v2/factscore_evidence' / key
    record_path = directory / 'record.json'
    session = None

    def get(params):
        response = session.get(API, params={'format': 'json', 'formatversion': 2, 'maxlag': 5, **params}, timeout=(20, 90))
        response.raise_for_status()
        result = response.json()
        if 'error' in result:
            raise ValueError(f'Wikipedia API error: {result["error"]}')
        return result, response.url

    try:
        if record_path.exists():
            record = json.loads(record_path.read_text(encoding='utf-8'))
            verify_record(root, topic, record)
            return record
        session = requests.Session()
        session.headers['User-Agent'] = 'AyaBaselineResearch/2.0 (public Wikipedia evidence snapshot)'
        session.mount('https://', HTTPAdapter(max_retries=Retry(
            total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=['GET'], respect_retry_after_header=True,
        )))
        metadata_path = directory / 'query.json'
        if metadata_path.exists():
            query_record = json.loads(metadata_path.read_text(encoding='utf-8'))
        else:
            query, query_url = get({'action': 'query', 'titles': topic, 'redirects': 1,
                                    'prop': 'revisions|info|pageprops', 'rvprop': 'ids|timestamp', 'inprop': 'url'})
            query_record = {'url': query_url, 'fetched_at_utc': datetime.now(timezone.utc).isoformat(), 'response': query}
            immutable(metadata_path, json_bytes(query_record))
        page = resolved_page(query_record, topic)
        revision = page['revisions'][0]
        oldid = revision['revid']
        parse_path = directory / 'parse.json'
        if parse_path.exists():
            parse_record = json.loads(parse_path.read_text(encoding='utf-8'))
            parsed = parse_record['response']
        else:
            parsed, parse_url = get({'action': 'parse', 'oldid': oldid, 'prop': 'text|revid', 'disableeditsection': 1})
            immutable(parse_path, json_bytes({'url': parse_url, 'fetched_at_utc': datetime.now(timezone.utc).isoformat(), 'response': parsed}))
        article = parsed['parse']
        if article.get('revid') != oldid or article.get('pageid') != page['pageid']:
            raise ValueError('Parsed page/revision does not match frozen query')
        text = clean_html(article['text'])
        if len(text.strip()) < 100:
            raise ValueError('Evidence article is unexpectedly empty/short')
        text_path = directory / 'text.txt'
        immutable(text_path, text.encode('utf-8'))
        files = [{'path': p.relative_to(root).as_posix(), 'bytes': p.stat().st_size,
                  'sha256': digest(p.read_bytes())} for p in [metadata_path, parse_path, text_path]]
        record = {'topic': topic, 'status': 'ok', 'title': page['title'], 'pageid': page['pageid'],
                  'revision': oldid, 'revision_timestamp': revision['timestamp'],
                  'url': f'https://en.wikipedia.org/w/index.php?oldid={oldid}',
                  'extractor_version': EXTRACTOR, 'files': files}
        immutable(record_path, json_bytes(record))
        verify_record(root, topic, record)
        return record
    except (requests.RequestException, ValueError, KeyError, IndexError, OSError) as exc:
        failure = {'topic': topic, 'status': 'retrieval_error', 'error': type(exc).__name__, 'message': str(exc)}
        # Preserve failed attempts; no fabricated evidence, replacement topic or score.
        error_path = directory / ('error-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '.json')
        immutable(error_path, json_bytes(failure))
        return failure
    finally:
        if session is not None:
            session.close()


def verify_record(root, topic, record):
    if record.get('topic') != topic or record.get('status') != 'ok' or record.get('extractor_version') != EXTRACTOR:
        raise ValueError(f'Invalid frozen record: {topic}')
    directory = root / 'data/v2/factscore_evidence' / digest(topic.encode('utf-8'))[:20]
    expected_paths = {p.relative_to(root).as_posix() for p in [directory/'query.json', directory/'parse.json', directory/'text.txt']}
    if {f['path'] for f in record['files']} != expected_paths or len(record['files']) != 3:
        raise ValueError('Unexpected evidence file inventory')
    for artifact in record['files']:
        data = (root / artifact['path']).read_bytes()
        if digest(data) != artifact['sha256'] or len(data) != artifact['bytes']:
            raise ValueError(f'Evidence hash mismatch: {artifact["path"]}')
    query_record = json.loads((directory / 'query.json').read_text(encoding='utf-8'))
    parse_record = json.loads((directory / 'parse.json').read_text(encoding='utf-8'))
    parsed = parse_record['response']['parse']
    page = resolved_page(query_record, topic)
    parse_url = urlparse(parse_record['url'])
    params = parse_qs(parse_url.query)
    if (parse_url.scheme != 'https' or parse_url.netloc != 'en.wikipedia.org' or parse_url.path != '/w/api.php'
            or params.get('oldid') != [str(record['revision'])] or params.get('action') != ['parse']):
        raise ValueError('Parse URL does not identify the frozen revision')
    if record['revision'] != page['revisions'][0]['revid'] or record['revision'] != parsed['revid']:
        raise ValueError('Revision mismatch in frozen evidence')
    if record['pageid'] != page['pageid'] or record['pageid'] != parsed['pageid'] or record['title'] != page['title']:
        raise ValueError('Topic-page identity mismatch')
    if (record['revision_timestamp'] != page['revisions'][0]['timestamp']
            or record['url'] != f'https://en.wikipedia.org/w/index.php?oldid={record["revision"]}'):
        raise ValueError('Revision timestamp/URL mismatch')
    if (directory / 'text.txt').read_bytes() != clean_html(parsed['text']).encode('utf-8'):
        raise ValueError('Derived text does not match preserved HTML')
    if len(clean_html(parsed['text']).strip()) < 100:
        raise ValueError('Evidence article is unexpectedly empty/short')


def reuse_record(root, topic):
    """Strictly verify an existing record. No HTTP, fetching, or failure artifacts."""
    path = root/'data/v2/factscore_evidence'/digest(topic.encode('utf-8'))[:20]/'record.json'
    content = path.read_bytes()  # Missing/corrupt records fail; there is no fallback.
    record = json.loads(content)
    if not isinstance(record, dict):
        raise ValueError(f'Invalid frozen record object: {topic}')
    try:
        verify_record(root, topic, record)
    except (KeyError, TypeError, IndexError, AttributeError) as error:
        raise ValueError(f'Invalid frozen record structure: {topic}') from error
    if content != json_bytes(record):
        raise ValueError('Per-topic record is not in its frozen serialization')
    return record


def same_json(actual, expected):
    # Compare JSON types too: False/0 and 0.0/0 must not be treated as equal indices.
    return json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True)


def evaluation_binding(root, names):
    selected, selection = subject_selection(root)
    if names != [topic for _, topic in selected] or len(names) != SELECTED_COUNT:
        raise ValueError('Evaluation biography subjects do not match the pinned source selection')
    path = root / 'data/v2/evaluation.jsonl'
    content = path.read_bytes()
    rows = [json.loads(line) for line in content.decode('utf-8').splitlines() if line.strip()]
    bios = [r for r in rows if r['dataset'] == 'factscore_bios']
    if len(bios) != SELECTED_COUNT:
        raise ValueError('Evaluation biography inventory does not match the pinned source selection')
    for row, (index, topic) in zip(bios, selected, strict=True):
        metadata = row.get('metadata', {})
        if (metadata.get('topic') != topic or type(metadata.get('source_index')) is not int
                or metadata['source_index'] != index or row.get('id') != f'v2-factscore_bios-{index}'
                or row.get('question') != PROMPT_TEMPLATE.format(topic=topic)
                or metadata.get('source_path') != FACTSCORE_PATH or metadata.get('zip_member') != FACTSCORE_MEMBER):
            raise ValueError('Evaluation biography subjects/indices/IDs/prompts do not match the pinned source selection')
    manifest_path = root / 'data/v2/manifest.json'
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if (not same_json(manifest.get('selection', {}).get('factscore_bios'), selection)
            or manifest.get('counts', {}).get('factscore_bios') != SELECTED_COUNT
            or manifest.get('total_evaluation_rows') != 9527 or manifest.get('new_generation_rows') != 7549
            or manifest.get('reused_rows_per_model') != 1978):
        raise ValueError('Evaluation manifest FActScore selection/counts changed')
    return {'path': path.relative_to(root).as_posix(), 'sha256': digest(content),
            'factscore_ids': [r['id'] for r in bios], 'factscore_source_indices': [index for index, _ in selected],
            'selection': selection,
            'manifest': {'path': manifest_path.relative_to(root).as_posix(), 'sha256': digest(manifest_bytes)}}


def evidence_report(records, selection):
    return {'version': EXTRACTOR, 'knowledge_source': 'English Wikipedia per-page oldid revisions frozen before Aya v2 generation',
            'deviation': 'Current pinned page revisions, not the original FActScore 2023-04-01 database. Wikipedia-backed estimate, not human answer-key matching.',
            'extraction_rules': EXTRACTION_RULES, 'source_archive_sha256': ARCHIVE_SHA256,
            'selection': selection, 'count': len(records), 'ready': all(r['status'] == 'ok' for r in records),
            'records': records}


def finalize_offline(root):
    """Publish a manifest only after reusing all 180 valid record.json files.

    This path never calls fetch_one or creates a session, and never writes errors
    or incomplete reports. An existing manifest is verified, never replaced.
    """
    selected, selection = subject_selection(root)
    names = [topic for _, topic in selected]
    path = root / 'data/v2/factscore-evidence-manifest.json'
    if path.exists():
        return verify(root)
    binding = evaluation_binding(root, names)
    records = [reuse_record(root, topic) for topic in names]
    report = evidence_report(records, selection)
    report['evaluation'] = binding
    immutable(path, json_bytes(report))
    return verify(root)


def freeze(root):
    selected, selection = subject_selection(root)
    names = [topic for _, topic in selected]
    path = root / 'data/v2/factscore-evidence-manifest.json'
    if path.exists():
        return verify(root)
    # Reject stale/reindexed evaluation before any retrieval can be attempted.
    binding = evaluation_binding(root, names)
    with ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(lambda topic: fetch_one(root, topic), names))
    errors = [r for r in records if r['status'] != 'ok']
    report = evidence_report(records, selection)
    if errors:
        failed_path = root / 'data/v2/factscore_evidence' / ('incomplete-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '.json')
        immutable(failed_path, json_bytes(report))
        raise RuntimeError(f'{len(errors)} evidence sources unavailable; incomplete report: {failed_path}')
    report['evaluation'] = binding
    immutable(path, json_bytes(report))
    return verify(root)


def verify(root):
    selected, selection = subject_selection(root)
    names = [topic for _, topic in selected]
    report = json.loads((root/'data/v2/factscore-evidence-manifest.json').read_text(encoding='utf-8'))
    if report.get('version') != EXTRACTOR or report.get('extraction_rules') != EXTRACTION_RULES:
        raise ValueError('Evidence extraction contract changed')
    if not report.get('ready') or report['count'] != SELECTED_COUNT or [r['topic'] for r in report['records']] != names:
        raise ValueError('Incomplete or reordered evidence manifest')
    if not same_json(report.get('selection'), selection):
        raise ValueError('Evidence manifest subject selection changed')
    if report['source_archive_sha256'] != ARCHIVE_SHA256:
        raise ValueError('Evidence subjects source changed')
    if not same_json(report.get('evaluation'), evaluation_binding(root, names)):
        raise ValueError('Evidence manifest evaluation binding changed')
    for topic, record in zip(names, report['records']):
        if json_bytes(reuse_record(root, topic)) != json_bytes(record):
            raise ValueError('Per-topic record differs from evidence manifest')
    return {'status': 'verified', 'subjects': len(names), 'files_hashed': 3*len(names), 'paid_calls': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['freeze', 'verify', 'finalize-offline'])
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps({'freeze': freeze, 'verify': verify, 'finalize-offline': finalize_offline}[args.command](args.root), indent=2))
