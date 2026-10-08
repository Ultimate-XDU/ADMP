```markdown
# 后端切片链条问题说明与修改方案

## 1. 现状说明

当前代码里的 chunk 确实已经有链条。

在 `BackEnd/aerospace_search_main.py` 中，`build_chain()` 会给每个 chunk 写入：

```python
prev_id
next_id
parent_id
parent_heading
heading_path
clause_path
doc_order
```

也就是说，目前系统已经有：

- 前后相邻 chunk 链条
- 父标题链条
- 标题路径
- 条款路径
- 文档内顺序

这些不是没有。

## 2. 现有链条缺的是什么

当前问题不是“没有链条”，而是：

> 链条连接的是 chunk，不是法规天然结构单元。

现在的 chunk 大多来自 PDF text block 或语义合并后的文本块。  
但法规/适航文件天然结构是：

```text
文档
  章
    节
      条
        款
          项
```

当前 chunk 链条无法稳定表达：

```text
第29.301条 > (a)
第21.171条 > （一）
第21.171条 > （二）
第二百一十条 > 施行日期
2．适用范围 > 正文
```

## 3. 当前链条和目标链条的区别

### 当前链条

```text
chunk_001 -> chunk_002 -> chunk_003
```

每个 chunk 有：

```python
content = "一段文本"
heading_path = ["C章 强度要求", "第一节 一般规定", "第29.301条 载荷"]
parent_id = "上一个标题 chunk"
next_id = "下一个 chunk"
```

问题是：`chunk_002` 通常不会在句中硬切，但它仍可能跨越多个法规结构单元，例如同时包含多个款、项，或者只是某个条/款下面的一段正文，而不是一个明确的 article/clause/item 节点。
对问答来说，我们更希望它们是三个可检索结构节点。

### 目标链条

```text
doc
  chapter: C章 强度要求
    section: 第一节 一般规定
      article: 第29.301条 载荷
        clause: (a)
        clause: (b)
        clause: (c)
```

每个节点都是一个明确结构单元，而不是随 PDF block 切出来的文本块。

## 4. 具体缺口

### 4.1 缺少 record_type

当前 chunk 只有大致的 `chunk_role`，例如：

```python
text
heading
requirement
definition
classification
```

但没有稳定的结构类型：

```python
doc
chapter
section
article
clause
item
paragraph
toc
```

所以系统不知道某个 chunk 是：

```text
第29.301条
```

还是：

```text
第29.301(a)
```

还是普通正文。

### 4.2 缺少 clause 级节点

例如正确答案在：

```text
第29.301条 载荷
(a）强度的要求用限制载荷和极限载荷来规定。
```

当前系统通常只知道：

```python
section_title = "第29.301 条 载荷"
content = "(a）强度的要求..."
```

但没有明确生成：

```python
article_label = "第29.301条"
article_title = "载荷"
clause_label = "(a)"
clause_id = "29.301(a)"
```

这会导致检索时不能精确命中 `29.301(a)`。

### 4.3 heading_path 是文本路径，不是结构树路径

当前：

```python
heading_path = ["C章 强度要求", "第一节 一般规定", "第29.301条 载荷"]
```

目标：

```python
heading_path = [
  {"type": "chapter", "label": "C章", "title": "强度要求"},
  {"type": "section", "label": "第一节", "title": "一般规定"},
  {"type": "article", "label": "第29.301条", "title": "载荷"},
  {"type": "clause", "label": "(a)", "title": ""}
]
```

文本路径能显示，但不能稳定计算父子关系、过滤目录页、精确检索款项。

### 4.4 parent_id 指向 chunk，不一定指向结构父节点

当前 `parent_id` 可能指向最近 heading chunk。

但目标中：

```text
clause (a) 的 parent_id 应该指向 article 第29.301条
article 第29.301条 的 parent_id 应该指向 section 第一节
section 第一节 的 parent_id 应该指向 chapter C章
```

现在的 parent_id 不是严格结构父子关系。

### 4.5 目录页和正文页混在一起

目录中的：

```text
C章 强度要求
第二节 飞行载荷
```

和正文中的：

```text
C章 强度要求
第一节 一般规定
第29.301条 载荷
```

目前都可能进入普通检索。

目标应该标记：

```python
record_type = "toc"
is_toc = True
```

默认普通 QA 不优先检索目录页。

### 4.6 跨页条款没有合并为同一结构记录

当前靠：

```python
next_id
prev_id
```

补上下文。

但目标应该是：

```python
article_id = "21.171"
page_start = 62
page_end = 63
body_text = "完整跨页条款内容"
```

也就是说，跨页内容应该在切片阶段合并成完整 article/clause record，而不是问答阶段临时追 next_id。

## 5. 为什么现在会出错

### 示例 1：适用范围问题

用户问：

```text
2024年12月6日下发的航空器的持续适航文件的适用范围
```

目标应该命中：

```text
2．适用范围
```

但旧切片可能把它放在：

```text
1．依据和目的 的正文块里
```

原因：

- PDF block 里包含多个标题
- 旧逻辑只看第一行是不是标题
- 后续 `2．适用范围` 没被切成独立结构节点

### 示例 2：强度要求问题

用户问：

```text
运输类旋翼航空器适航规定中的强度要求由什么来规定
```

目标应该命中：

```text
C章 强度要求 > 第一节 一般规定 > 第29.301条 载荷 > (a)
```

但当前系统可能命中：

```text
第29.519条
第29.549条
第29.573条
```

原因：

- 这些条款正文也包含“必须设计成能承受载荷”
- 当前检索不知道 `第29.301(a)` 是 `C章 强度要求` 下的总则条款
- 款 `(a)` 没有成为一等检索节点

## 6. 修改目标

切片阶段要输出两种数据：

### 6.1 结构记录 StructuralRecord

```python
{
  "record_id": "...",
  "doc_name": "运输类旋翼航空器适航规定.pdf",
  "doc_title": "运输类旋翼航空器适航规定",
  "record_type": "clause",
  "label": "(a)",
  "title": "",
  "article_label": "第29.301条",
  "article_title": "载荷",
  "chapter_label": "C章",
  "chapter_title": "强度要求",
  "section_label": "第一节",
  "section_title": "一般规定",
  "heading_path": "运输类旋翼航空器适航规定 > C章 强度要求 > 第一节 一般规定 > 第29.301条 载荷 > (a)",
  "body_text": "强度的要求用限制载荷（使用中预期的最大载荷）和极限载荷（限制载荷乘以规定的安全系数）来规定。",
  "page_start": 33,
  "page_end": 33,
  "parent_id": "article_29.301",
  "prev_id": "...",
  "next_id": "...",
  "order": 123
}
```

### 6.2 检索记录 RetrievalRecord

从结构记录派生：

```python
{
  "record_id": "...",
  "metadata_text": "运输类旋翼航空器适航规定 > C章 强度要求 > 第一节 一般规定 > 第29.301条 载荷 > (a)",
  "body_text": "强度的要求用限制载荷和极限载荷来规定。",
  "retrieval_text": "运输类旋翼航空器适航规定 > C章 强度要求 > 第一节 一般规定 > 第29.301条 载荷 > (a)\n强度的要求用限制载荷和极限载荷来规定。"
}
```

## 7. 建议修改步骤

### Step 1：新增 parse_heading()

统一识别标题和结构标签。

输入：

```python
line: str
```

输出：

```python
{
  "type": "article",
  "level": 3,
  "label": "第29.301条",
  "title": "载荷"
}
```

需要支持：

```text
A 章 总则
C 章 强度要求
第七章 适航证、适航批准标签和外国适航证认可书
第一节 一般规定
第29.301 条 载荷
第29.301条
29.301 载荷
第二百一十条 本规定自2026年7月1日起施行
(a）强度的要求...
(b）除非另有说明...
(1）...
（1）...
（一）...
1．依据和目的
2．适用范围
```

### Step 2：PDF block 内二次切分

不要只看 block 第一行。

如果一个 block 是：

```text
1．依据和目的
...
2．适用范围
...
3．撤销
...
```

必须切成：

```text
1．依据和目的 + body
2．适用范围 + body
3．撤销 + body
```

### Step 3：构建结构树

维护结构栈：

```python
doc_stack
chapter_stack
section_stack
article_stack
clause_stack
```

遇到新标题时，根据类型挂载：

```text
chapter -> section -> article -> clause -> item
```

### Step 4：生成 article/clause/item records

不要只生成 chunk。

例如：

```text
第29.301条 载荷
(a）...
(b）...
(c）...
```

输出：

```text
record_type=article, label=第29.301条, title=载荷
record_type=clause, label=(a), parent=第29.301条
record_type=clause, label=(b), parent=第29.301条
record_type=clause, label=(c), parent=第29.301条
```

### Step 5：跨页合并

如果一个条款跨页：

```text
第21.171条 在第62页开始
内容在第63页结束
```

输出一个完整 article record：

```python
page_start = 62
page_end = 63
body_text = "完整条款内容"
```

### Step 6：目录页标记

识别目录页：

```python
is_toc = True
record_type = "toc"
```

普通 QA 默认降低目录页权重。

### Step 7：保留旧 chunk 兼容

不要直接删旧 chunk。

建议并行输出：

```text
旧 chunk index
新 structural index
```

检索层先 A/B 测试。


```

## 8. 最小验收标准

下面这些必须能在切片结果里直接找到独立 record：

```text
航空器的持续适航.pdf
  2．适用范围

运输机场运行安全管理规定.pdf
  第二百一十条 本规定自2026年7月1日起施行

运输类旋翼航空器适航规定.pdf
  C章 强度要求
    第一节 一般规定
      第29.301条 载荷
        (a）强度的要求用限制载荷和极限载荷来规定

民用航空产品和零件合格审定规定.pdf
  第21.171条 适航证的类别
    （一）标准适航证...
    （二）特殊适航证...
```

## 10. 总结

当前 chunk 有链条，但链条粒度不对。

当前是：

```text
chunk 链条
```

目标是：

```text
法规结构链条
```

也就是：

```text
文档 -> 章 -> 节 -> 条 -> 款 -> 项
```

只有把“款”和“项”变成一等结构节点，后面的 ES、Milvus、Reranker 才能稳定工作。
```

## 11. 当前工程实现状态（2026-04-28）

已完成：

- 新增 `aerospace_structure.py`，结构树构建逻辑从原检索主链路中独立出来。
- 结构记录已支持 `doc/chapter/section/article/clause/item/toc` 等类型。
- 结构记录已写入 `parent_record_id`、`child_record_ids`、`direct_child_count`、`descendant_count`、`heading_path`、`page_start/page_end`。
- 已区分目录页和正文页，普通 QA 默认跳过 TOC 结构记录。
- 已支持 `5 > 5.1 > 5.1.1` 这类数字层级的父子关系，并能补齐缺失的中间父节点。
- 已支持条、款、项作为一等结构记录进入检索候选。
- 已增加 `DocumentProfile` 和动态 marker 栈：先统计文档内 `alpha/chinese -> digit -> roman` 等编号层级，再把 `(a)/(1)/(ⅰ)` 自动挂载为父子结构，避免按单个样式打补丁。
- 索引阶段会把结构树标记回填到旧 chunk，保留旧 chunk 兼容，同时增加结构化检索字段。
- QA 阶段已优先处理结构计数和定义类问题，避免“全文最大条号”抢答章节计数。
- QA 证据已增加质量门：只把高质量证据送入 prompt 和前端 sources，弱语义证据不再默认混入。
- 结构证据已接入 PDF 定位修复，尽量补齐页码、高亮文本和 bbox。

已验证问题：

```text
运输机场运行安全管理规定总则共有几条 -> 8 条
设计批准持有人系统要求有几条 -> 3 个直接小节
系统建设要素有几个 -> 9 个直接小节
持续适航的定义是什么 -> 命中 2.1 持续适航
运输机场运行安全管理规定是在什么时候开始施行的 -> 2026年7月1日
航空器的持续适航文件的适用范围 -> 命中 2．适用范围
运输类旋翼航空器适航规定中的强度要求由什么来规定 -> 命中第29.301条(a)
民用航空产品和零件合格审定规定第21.171条适航证的类别有哪些 -> 命中第21.171条及（一）（二）
运输类旋翼航空器的动力装置包括哪些部件 -> 命中第29.901条(a)及其(1)(2)(3)
用于直升机的活塞发动机必须符合哪项规定 -> 命中第29.903条(a)
燃油箱试验中，振动试验频率需要遵循哪些规定 -> 命中第29.965条(d)(3)
```

暂未展开：

- 复合问题拆解、多跳检索、跨文档链式推理。
- 关系图谱/知识图谱。当前阶段先不引入 KG，先把结构树和证据链路打稳。
- 全量前端证据交互优化。当前后端已尽量提供更准确的 source、highlight 和 bbox。
