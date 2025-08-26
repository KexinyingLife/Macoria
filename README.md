# 📦 Macoria

**Macoria** 是一个基于 SwiftUI 开发的 macOS 应用，用于 **检索 GitHub 项目、快速下载与管理**，支持收藏与个性化推荐，界面风格贴近原生 macOS。  

---

## ✨ 功能特性
- 🔍 **项目检索**  
  - 支持关键字搜索 GitHub 项目  
  - 支持按推荐条件自动检索（默认 `topic:macos stars:>100 pushed:>2020-01-01 archived:false`）  

- ⭐ **收藏功能**  
  - 一键收藏常用项目  
  - 收藏项目会优先显示在推荐页顶部  

- ⬇️ **下载管理**  
  - 顶部菜单栏与详情页同步下载进度  
  - 支持“重新下载”与重名文件自动编号 `(1)(2)…`  

- ⚙️ **个性化推荐**  
  - 可在设置界面选择推荐参数（Stars 数、是否包含 Fork、更新时间等）  
  - 空搜索时自动进入推荐模式  

- 🌐 **多语言支持**  
  - 提供 **中文** 与 **英文** UI 自动切换  
  - 所有提示语、菜单、快捷键均已本地化  

- 🎨 **原生 macOS 设计**  
  - 使用 SwiftUI 打造，贴合 macOS 14+ 风格  
  - 设置界面采用 macOS 系统偏好设置风格  

---


## ⌨️ 快捷键 & 菜单
- `⌘,` 打开设置  
- `⌘Q` 退出  
- `⌘R` 刷新当前检索  
- `⌘W` 关闭窗口  
- `⌘C / ⌘V` 复制、粘贴  
- 更多功能请查看菜单栏  

---

## 🛠 技术栈
- **Swift 6 + SwiftUI**  
- **URLSession** 用于 GitHub API 请求  
- **Combine** 实现数据驱动界面更新  
- **AppStorage / UserDefaults** 持久化用户配置  

---

## 🚀 开始使用
```bash
git clone https://github.com/KexinyingLife/Macoria.git
cd Macoria
open Macoria.xcodeproj
```
在 Xcode 中运行即可。  

---

## 📜 License
本项目采用 [MIT License](LICENSE)。  
