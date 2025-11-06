import { execFile } from "child_process";

// on macOS, use Launch Services API

interface NormalizeUrlOptions {
  /** Scheme to prepend when missing. */
  defaultScheme?: 'https' | 'http';
  /** Prepend 'www.' for bare domains (ignored for IP/localhost). */
  addWWW?: boolean;
  /** Base URL for resolving relative paths (string or URL). */
  base?: string | URL;
}

const SCHEME_RE = /^[a-zA-Z][a-zA-Z0-9+.-]*:/;
const EMAIL_RE  = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
const DOMAIN_RE = /^(?:[a-z0-9-]+\.)+[a-z]{2,63}(?::\d{2,5})?(?:\/\S*)?$/i;
const IPV4_RE   = /^(?:\d{1,3}\.){3}\d{1,3}(?::\d{2,5})?(?:\/\S*)?$/;
const LOCAL_RE  = /^localhost(?::\d{2,5})?(?:\/\S*)?$/i;

function normalizeUrl(
  input: string | null | undefined,
  opts: NormalizeUrlOptions = {}
): string | null {
  const { defaultScheme, addWWW, base } = normalizeOptions(opts);

  if (typeof input !== 'string') return null;

  let s = input.trim();
  if (!s) return null;

  // Strip common surrounding punctuation and trailing punctuation
  if (
    (s.startsWith('<') && s.endsWith('>')) ||
    (s.startsWith('"') && s.endsWith('"')) ||
    (s.startsWith("'") && s.endsWith("'"))
  ) {
    s = s.slice(1, -1).trim();
  }
  s = s.replace(/[)\].,!?:;]+$/, '');

  // Protocol-relative → add scheme
  if (s.startsWith('//')) s = `${defaultScheme}://${s.slice(2)}`;

  const hasScheme = SCHEME_RE.test(s);

  if (!hasScheme) {
    // Email → mailto:
    if (EMAIL_RE.test(s)) return `mailto:${s}`;

    // Relative path → absolutize if base provided
    if (s.startsWith('/') && base) {
      try {
        return new URL(s, base).toString();
      } catch {
        return null;
      }
    }

    // Bare host/IP/localhost → prepend scheme (+ optional www)
    const looksDomain = DOMAIN_RE.test(s);
    const looksIPv4   = IPV4_RE.test(s);
    const isLocal     = LOCAL_RE.test(s);

    if (looksDomain || looksIPv4 || isLocal) {
      if (addWWW && looksDomain && !/^www\./i.test(s)) s = `www.${s}`;
      s = `${defaultScheme}://${s}`;
    } else {
      return null;
    }
  }

  // Final parse & canonicalization
  try {
    const resolvedBase = base ? (base instanceof URL ? base : new URL(String(base))) : undefined;
    const url = new URL(s, resolvedBase);
    url.protocol = url.protocol.toLowerCase();
    url.hostname = url.hostname.toLowerCase();
    if ((url.protocol === 'http:' && url.port === '80') ||
        (url.protocol === 'https:' && url.port === '443')) {
      url.port = '';
    }
    return url.toString();
  } catch {
    return null;
  }
}

// Optional helper: quick “is this URL-like?” check (no scheme required)
function isUrlLike(s: string): boolean {
  if (typeof s !== 'string') return false;
  const t = s.trim();
  if (!t) return false;
  if (t.startsWith('//') || SCHEME_RE.test(t)) return true;
  if (EMAIL_RE.test(t)) return true;
  return DOMAIN_RE.test(t) || IPV4_RE.test(t) || LOCAL_RE.test(t) || t.startsWith('/');
}

function normalizeOptions(opts: NormalizeUrlOptions): NormalizeUrlOptions {
  const defaultScheme = opts.defaultScheme ?? 'https';
  return {
    defaultScheme: defaultScheme,
    addWWW: !!opts.addWWW,
    base: opts.base,
  };
}

export default async function open(things: string | Array<string>): Promise<void> {
  const thingList = Array.isArray(things) ? things : [things];
  let normalizedThings: string[] = [];
  for (const thing of thingList) {
    const normalizedThing = normalizeUrl(thing);
    normalizedThings.push(normalizedThing ?? thing);
  }

  return new Promise((resolve, reject) => {
    execFile("/usr/bin/open", normalizedThings, (error, stdout, stderr) => {
      if (error) {
        reject(new Error(stderr || error.message));
      } else {
        resolve();
      }
    });
  });
}
