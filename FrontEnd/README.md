# Aviation Knowledge Command

这是一个基于 Vite + React + TypeScript 的前端项目，现已整理为可在 Windows 和 macOS 上继续打开、运行和修改的源码版本。

## 环境要求

- Node.js 18 或更高版本
- npm 9 或更高版本

## 开发启动

```bash
npm run dev
```

项目会在启动前自动检查当前系统平台。如果你是把整个文件夹从 Windows 拷到 macOS，脚本会自动重新执行 `npm install`，补齐 macOS 所需依赖。

## macOS 使用

1. 把整个项目拷到 Mac。
2. 用 VS Code、WebStorm 或其他编辑器直接打开项目文件夹。
3. 第一次运行前，在终端进入项目目录后执行一次：

```bash
chmod +x start.command
```

4. 之后就可以直接在 Finder 里双击 `start.command` 启动。

启动后会自动打开浏览器到本地页面。你修改 `src` 里的代码后：

- 保存文件，页面会自动刷新。
- 如果你更习惯“关掉再双击打开一次”，也可以，重新启动后看到的就是最新页面。

如果你不想用 `start.command`，也可以直接执行：

```bash
npm run dev
```

## 常用命令

```bash
npm run dev
npm run build
npm run preview
npm run serve:dist
```

## 如果从 Windows 拷到 Mac 后仍然报依赖错误

执行下面这组命令即可：

```bash
rm -rf node_modules dist .platform-stamp.json
npm install
npm run dev
```
