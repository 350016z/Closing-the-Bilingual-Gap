
/**
 * Converts a Unicode Code Point index (used by Python/Backend) to a UTF-16 Code Unit index (used by JS/String.slice).
 * @param str The string to traverse.
 * @param cpIndex The target code point index.
 * @returns The corresponding UTF-16 index.
 */
export function toUTF16Index(str: string, cpIndex: number): number {
    let curCp = 0;
    let curUtf16 = 0;
    for (const char of str) {
        if (curCp === cpIndex) return curUtf16;
        curCp++;
        curUtf16 += char.length;
    }
    return curUtf16;
}

/**
 * Converts a UTF-16 Code Unit index (used by JS/Selection) to a Unicode Code Point index (used by Python/Backend).
 * @param str The string to traverse.
 * @param utf16Index The target UTF-16 index.
 * @returns The corresponding Code Point index.
 */
export function toCodePointIndex(str: string, utf16Index: number): number {
    let curCp = 0;
    let curUtf16 = 0;
    for (const char of str) {
        if (curUtf16 >= utf16Index) return curCp; // >= matches if inside a surrogate pair (shouldn't happen with valid alignment)
        curCp++;
        curUtf16 += char.length;
    }
    return curCp;
}
