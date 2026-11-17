# ADMP 航空知识工作台

React + TypeScript + Vite 前端。以问答、文档检索、知识库和原文核验为主，使用真实后端接口展示数据。

## 在 Ubuntu 启动

从项目根目录运行：

```bash
./scripts/admp-round2
```

页面位于 `http://localhost:5174/`，API 位于 `http://127.0.0.1:8788`。前端开发服务会将 `/api` 转发至该地址。也可以在 `FrontEnd` 目录设置 `VITE_API_PROXY=http://127.0.0.1:8788` 后单独运行 `npm run dev -- --port 5174`，或通过构建环境变量 `VITE_MANAGEMENT_API_BASE` 指定其他 API 地址。

项目启动脚本会检查平台依赖；跨 Windows、macOS 和 Linux 复制源码后，使用脚本重新安装相应平台的依赖即可。

## 工作区

- `/aviation-assistant`：带历史会话的文档问答。支持自动、单跳与多跳检索。
- `/trace`：仅检索，展示每跳查询、返回片段和原页。
- `/upload`：知识库与文档管理，支持 PDF、DOCX、PPTX、XLSX 上传、索引和真实状态反馈。
- `/graph`：文档结构、实体关系与来源追溯；语义图需要显式构建。
- `/system`：服务连通状态及使用说明。连通状态不等同于索引内容或检索质量。

图文增强默认关闭。开启后，后端将召回的文字证据与最多两张不同的 PDF 原页一起交给本地模型。页面会标明实际使用的页图或文本回退状态。这一功能是文本召回后的视觉问答，不是视觉向量检索。

答案引用通过后端 `citations[].id → sourceIndex` 映射定位，支持非连续引用编号。原文抽屉展示 PDF 页码、高亮和完整检索上下文。Markdown 表格与代码块使用标准渲染器，代码中的方括号数字不会被当成引用。

## 检查

```bash
npm run build
npm run test:citations
```

构建先运行 TypeScript，再生成 Vite 产物。引用检查编译并运行真实 React 组件，验证加粗、表格、非连续编号、代码块、普通链接和实际点击回调。它不模拟后端，也不代替实际问答验收。

## 源码入口

- `src/App.tsx`：导航、历史会话和工作区状态。
- `src/api.ts`：HTTP 入口、接口类型、上传与索引流程。
- `src/components/Workspace.tsx`：问答、检索、请求状态与图文选项。
- `src/components/LibraryView.tsx`：知识库及文档管理。
- `src/components/AnswerText.tsx`：Markdown 和引用映射。
- `src/components/EvidencePanel.tsx`：原页与证据上下文。
- `src/components/RetrievalSteps.tsx`：真实检索步骤与补检状态。

样式提供桌面和移动布局、轻量指针响应、焦点与悬停反馈，并遵循系统的减少动态效果设置。
