import assert from "node:assert/strict";
import { createElement, isValidElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { AnswerText } from "../src/components/AnswerText";

const sources = [
  { doc_name: "NASA.pdf", chunk_id: "nasa-page-2", page_number: 2 },
  { doc_name: "CAAC.pdf", chunk_id: "caac-page-8", page_number: 8 },
];
const citations = [
  { id: 2, sourceIndex: 1, chunkId: "caac-page-8" },
  { id: 9, sourceIndex: 0, chunkId: "nasa-page-2" },
];

function renderedButtons(element: unknown): any[] {
  if (Array.isArray(element)) return element.flatMap(renderedButtons);
  if (!isValidElement(element)) return [];
  const node = element as any;
  if (typeof node.type === "function")
    return renderedButtons(node.type(node.props));
  return [
    ...(node.type === "button" ? [node] : []),
    ...renderedButtons(node.props.children),
  ];
}

export function runChecks() {
  const checks: { name: string; status: string }[] = [];
  const cases = [
    {
      name: "citation remains clickable inside bold text",
      text: "**结冰影响升力 [2]**",
      expected: ["CAAC.pdf"],
      verify: (html: string) =>
        assert.match(
          html,
          /<strong>结冰影响升力 <button[^>]*aria-label="查看引用 2"/,
        ),
    },
    {
      name: "GFM table renders with clickable evidence citation",
      text: "| 现象 | 证据 |\n| --- | --- |\n| 升力下降 | [9] |",
      expected: ["NASA.pdf"],
      verify: (html: string) => {
        assert.match(html, /<table>/);
        assert.match(html, /<thead>/);
        assert.match(html, /<tbody>/);
        assert.match(html, /<td><button[^>]*aria-label="查看引用 9"/);
      },
    },
    {
      name: "sparse citation IDs select mapped source indexes",
      text: "先看 [9]，再看 [2]；未映射的 [1] 保留为文字。",
      expected: ["NASA.pdf", "CAAC.pdf"],
      verify: (html: string) => assert.match(html, /未映射的 \[1\] 保留为文字/),
    },
    {
      name: "inline code preserves bracketed numbers",
      text: "`array[9]` 与正文 [2]",
      expected: ["CAAC.pdf"],
      verify: (html: string) => assert.match(html, /<code>array\[9\]<\/code>/),
    },
    {
      name: "fenced code preserves bracketed numbers",
      text: "```text\n[9] and [2]\n```\n\n正文 [9]",
      expected: ["NASA.pdf"],
      verify: (html: string) =>
        assert.match(
          html,
          /<pre><code class="language-text">\[9\] and \[2\]\n<\/code><\/pre>/,
        ),
    },
    {
      name: "ordinary Markdown links are retained",
      text: "[NASA](https://www.nasa.gov/) 与 [2]",
      expected: ["CAAC.pdf"],
      verify: (html: string) =>
        assert.match(
          html,
          /<a href="https:\/\/www\.nasa\.gov\/"[^>]*>NASA<\/a>/,
        ),
    },
    {
      name: "repeated citations preserve all surrounding text",
      text: "甲 [9] 乙 [999] 丙 [2][9] 丁",
      expected: ["NASA.pdf", "CAAC.pdf", "NASA.pdf"],
      verify: (html: string) => {
        assert.match(html, /甲 /);
        assert.match(html, / 乙 \[999\] 丙 /);
        assert.match(html, / 丁/);
      },
    },
  ];
  for (const testCase of cases) {
    const selected: string[] = [];
    const props = {
      text: testCase.text,
      sources: sources as any,
      citations,
      onSelect: (source: any) => selected.push(source.doc_name),
    };
    const html = renderToStaticMarkup(createElement(AnswerText, props));
    testCase.verify(html);
    const buttons = renderedButtons(createElement(AnswerText, props));
    assert.equal(buttons.length, testCase.expected.length, testCase.name);
    for (const button of buttons) button.props.onClick();
    assert.deepEqual(selected, testCase.expected, testCase.name);
    checks.push({ name: testCase.name, status: "passed" });
  }
  return { checks, total: checks.length, passed: checks.length };
}
