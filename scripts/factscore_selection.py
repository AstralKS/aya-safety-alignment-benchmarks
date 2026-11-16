"""One offline, pinned FActScore selection shared by preparation and evidence.

The released 183-subject archive is unchanged. Only the three explicitly approved
local exclusions are omitted; retained rows keep their original indices and IDs.
"""
from __future__ import annotations

import hashlib
import io
import zipfile

FACTSCORE_PATH = 'data/v2/raw/FActScore/data.zip'
FACTSCORE_MEMBER = 'data/labeled/prompt_entities.txt'
ARCHIVE_SHA256 = '404482c9a7d588d581a2f7a78d609fe6bfec4ddc0f729c8560a61b1743dbffab'
MEMBER_SHA256 = '3e6cb69444cbe6621d155e4d65ef78498cb3803fe7899b911b7487bdd5768a7e'
SOURCE_RECEIPT_PATH = 'data/v2/new-sources-receipt.json'
SOURCE_RECEIPT_SHA256 = 'fdda39a0053bcab5988c64dae768e5004c3021d9ae93b594e9a44510420dcf65'
ORIGINAL_COUNT = 183
SELECTED_COUNT = 180
EXCLUSIONS = ((2, 'William Post'), (127, 'Carlos Alfonso'), (162, 'Jagan'))
EXCLUSION_REASON = 'Explicit user-approved local exclusion from the scored biography selection; raw source retained unchanged.'
PROMPT_TEMPLATE = 'Question: Tell me a bio of {topic}.'


def select_inventory(topics):
    """Validate the full original inventory before applying exact index/name pairs."""
    if (not isinstance(topics, list) or len(topics) != ORIGINAL_COUNT
            or any(not isinstance(topic, str) or not topic.strip() for topic in topics)
            or len(set(topics)) != ORIGINAL_COUNT):
        raise ValueError('FActScore must contain exactly 183 unique nonempty topics')
    if any(topics[index] != topic for index, topic in EXCLUSIONS):
        raise ValueError('FActScore approved exclusion names/source indices changed')
    excluded_indices = {index for index, _ in EXCLUSIONS}
    selected = [(index, topic) for index, topic in enumerate(topics) if index not in excluded_indices]
    if len(selected) != SELECTED_COUNT:
        raise ValueError('Unexpected FActScore selected count')
    return selected


def select_archive(data):
    """Verify fixed archive/member anchors and return rows plus selection metadata."""
    if hashlib.sha256(data).hexdigest() != ARCHIVE_SHA256:
        raise ValueError('FActScore source archive hash changed')
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        if archive.namelist().count(FACTSCORE_MEMBER) != 1:
            raise ValueError('FActScore prompt member missing/duplicated')
        content = archive.read(FACTSCORE_MEMBER)
    topics = content.decode('utf-8').splitlines()
    selected = select_inventory(topics)
    if hashlib.sha256(content).hexdigest() != MEMBER_SHA256:
        raise ValueError('FActScore prompt member hash changed')
    return selected, {
        'source_path': FACTSCORE_PATH, 'zip_member': FACTSCORE_MEMBER,
        'source_archive_sha256': ARCHIVE_SHA256, 'member_sha256': MEMBER_SHA256,
        'raw_count': ORIGINAL_COUNT, 'original_count': ORIGINAL_COUNT, 'selected_count': SELECTED_COUNT,
        'excluded_subjects': [{'topic': topic, 'source_index': index, 'reason': EXCLUSION_REASON}
                              for index, topic in EXCLUSIONS],
        'selected_source_indices': [index for index, _ in selected],
        'selected_ids': [f'v2-factscore_bios-{index}' for index, _ in selected],
        'sampling': 'original source order after exactly three user-approved local exclusions; no renumbering or prompt changes',
        'max_new_tokens': 1536, 'prompt_template': PROMPT_TEMPLATE, 'old_atomic_labels_used': False,
    }
