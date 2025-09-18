import { log } from "console";

export interface MetaData {
  num_words: number;
  max_len: number;
  index_from: number;
  start_id: number;
  oov_id: number;
  pad_id: number;
}

export type Vocab = Record<string, number>; // word -> index

export function normalizeText(str: string): string {
  return str
    .toLowerCase()
    .replace(/[^a-z0-9\s']/g, " ") // keep letters, numbers, spaces, apostrophes
    .replace(/\s+/g, " ")
    .trim();
}

export function textToSequence(
  text: string,
  vocab: Vocab,
  meta: MetaData
): number[] {
  const { num_words, max_len, index_from, start_id, oov_id, pad_id } = meta;

  const tokens = normalizeText(text).split(" ").filter(Boolean);

  const ids: number[] = [];
  ids.push(start_id);

  for (const t of tokens) {
    const baseIndex = vocab[t];
    let idx: number;
    if (baseIndex !== undefined) {
      idx = baseIndex + index_from;
      if (idx >= num_words) idx = oov_id;
    } else {
      idx = oov_id;
    }
    ids.push(idx);
    if (ids.length >= max_len) break; // truncating='post'
  }

  // padding='post'
  while (ids.length < max_len) ids.push(pad_id);

  return ids;
}
