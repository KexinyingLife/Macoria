//
//  MacoriaApp.swift
//  Macoria
//
//  Single-file SwiftUI macOS app that searches GitHub for macOS apps' latest releases
//  and surfaces assets like .dmg / mac*.zip / .pkg. It auto-filters by current CPU arch
//  (arm64/x86_64). It supports optional GitHub Personal Access Token input in-app.
//

import SwiftUI
import WebKit

struct MarkdownWebView: NSViewRepresentable {
    let markdown: String

    func makeNSView(context: Context) -> WKWebView {
        let webView = WKWebView()
        loadMarkdown(into: webView)
        return webView
    }

    func updateNSView(_ webView: WKWebView, context: Context) {
        loadMarkdown(into: webView)
    }

    private func loadMarkdown(into webView: WKWebView) {
        let css = """
        <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", Helvetica, Arial, sans-serif;
            padding: 16px;
            line-height: 1.6;
            color: inherit;
            background: transparent;
        }
        @media (prefers-color-scheme: dark) {
            body { color: #ddd; }
            pre, code { background: #2d2d2d; }
        }
        h1, h2, h3 { border-bottom: 1px solid #444; padding-bottom: .3em; }
        pre { padding: 12px; border-radius: 6px; overflow-x: auto; }
        code { font-family: monospace; padding: 2px 4px; border-radius: 4px; }
        a { color: #0366d6; text-decoration: none; }
        a:hover { text-decoration: underline; }
        </style>
        """

        let mdBase64 = Data(markdown.utf8).base64EncodedString()

        let html = """
        <html>
        <head>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        \(css)
        <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.8.0/styles/github.min.css">
        <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.8.0/highlight.min.js"></script>
        <script>
        marked.setOptions({
          highlight: function(code, lang) {
            if (hljs.getLanguage(lang)) {
              return hljs.highlight(code, { language: lang }).value;
            }
            return hljs.highlightAuto(code).value;
          }
        });
        </script>
        </head>
        <body>
        <div id="content">加载中...</div>
        <script>
        const md = atob("\(mdBase64)");
        document.getElementById('content').innerHTML = marked.parse(md);
        </script>
        </body>
        </html>
        """

        webView.setValue(false, forKey: "drawsBackground")
        webView.loadHTMLString(html, baseURL: nil)
    }
}
import Combine

// MARK: - Models

struct Repo: Codable, Identifiable, Hashable {
    let id: Int
    let name: String
    let fullName: String
    let description: String?
    let stargazersCount: Int
    let htmlURL: URL
    let owner: Owner
    
    enum CodingKeys: String, CodingKey {
        case id, name, owner, description
        case fullName = "full_name"
        case stargazersCount = "stargazers_count"
        case htmlURL = "html_url"
    }
    
    struct Owner: Codable, Hashable {
        let login: String
        let avatarURL: URL?
        enum CodingKeys: String, CodingKey {
            case login
            case avatarURL = "avatar_url"
        }
    }
}

struct Release: Codable, Identifiable, Equatable {
    let id: Int
    let tagName: String
    let name: String?
    let draft: Bool
    let prerelease: Bool
    let publishedAt: Date?
    let assets: [Asset]

    enum CodingKeys: String, CodingKey {
        case id, assets, draft, prerelease, name
        case tagName = "tag_name"
        case publishedAt = "published_at"
    }
}

struct Asset: Codable, Identifiable, Equatable {
    let id: Int
    let name: String
    let size: Int
    let browserDownloadURL: URL
    
    enum CodingKeys: String, CodingKey {
        case id, name, size
        case browserDownloadURL = "browser_download_url"
    }
}

// A unified item representing one app (repo + chosen asset set)
struct AppItem: Identifiable, Codable, Hashable, Equatable {
    let id: Int
    let repo: Repo
    let release: Release
    let assets: [Asset] // filtered for macOS or fallback (all assets)
    let readme: String?

    static func == (lhs: AppItem, rhs: AppItem) -> Bool {
        return lhs.id == rhs.id
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(id)
    }
}

// MARK: - CPU Arch

enum CPUArch: String {
    case arm64, x86_64
    static var current: CPUArch {
        #if arch(arm64)
        return .arm64
        #else
        return .x86_64
        #endif
    }
}

// MARK: - Settings Store

final class SettingsStore: ObservableObject {
    @AppStorage("githubToken") var githubToken: String = ""
    @AppStorage("recommendedQuery") var recommendedQuery: String =
        "topic:macos stars:>100 pushed:>2020-01-01 archived:false"
    @AppStorage("pageLimit") var pageLimit: Int = 2 // each page returns 30 repos
    @AppStorage("prereleaseAllowed") var prereleaseAllowed: Bool = false
}

// MARK: - Recommended Query Generator
// Helper to generate recommended query string from current settings.
extension SettingsStore {
    var recommendedQueryString: String {
        // Parse out relevant values from recommendedQuery or settings
        // Try to extract minStars, pushed date, includeForks, archived
        var minStars = 100
        var pushed = "2020-01-01"
        var includeForks = true
        // Try to parse from recommendedQuery string if possible
        let q = recommendedQuery
        if let starsRange = q.range(of: "stars:>") {
            let after = q[starsRange.upperBound...]
            let numStr = after.split(separator: " ").first ?? ""
            if let stars = Int(numStr) {
                minStars = stars
            }
        }
        if let pushedRange = q.range(of: "pushed:>") {
            let after = q[pushedRange.upperBound...]
            let dateStr = after.split(separator: " ").first ?? ""
            if !dateStr.isEmpty {
                pushed = String(dateStr)
            }
        }
        if q.contains("fork:false") {
            includeForks = false
        }
        var components: [String] = []
        components.append("topic:macos")
        components.append("stars:>\(minStars)")
        components.append("pushed:>\(pushed)")
        components.append("archived:false")
        if !includeForks {
            components.append("fork:false")
        }
        return components.joined(separator: " ")
    }
}

// MARK: - GitHub API Client

actor GitHubClient {
    private let token: String
    private let decoder: JSONDecoder
    
    init(token: String) {
        self.token = token.trimmingCharacters(in: .whitespacesAndNewlines)
        self.decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
    }
    
    private func request(_ url: URL) async throws -> (Data, HTTPURLResponse) {
        var req = URLRequest(url: url)
        req.setValue("application/vnd.github+json", forHTTPHeaderField: "Accept")
        req.setValue("Macoria/1.0", forHTTPHeaderField: "User-Agent")
        if !token.isEmpty {
            req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        let (data, resp) = try await URLSession.shared.data(for: req)
        guard let http = resp as? HTTPURLResponse else {
            throw URLError(.badServerResponse)
        }
        return (data, http)
    }
    
    func searchRepositories(query: String, page: Int = 1) async throws -> [Repo] {
        // GitHub search API
        // GET https://api.github.com/search/repositories?q=<q>&sort=stars&order=desc&page=<p>&per_page=30
        var comps = URLComponents(string: "https://api.github.com/search/repositories")!
        comps.queryItems = [
            .init(name: "q", value: query),
            .init(name: "sort", value: "stars"),
            .init(name: "order", value: "desc"),
            .init(name: "page", value: "\(page)"),
            .init(name: "per_page", value: "30"),
        ]
        let (data, http) = try await request(comps.url!)
        guard http.statusCode == 200 else {
            throw NSError(domain: "GitHub", code: http.statusCode, userInfo: [
                NSLocalizedDescriptionKey: "Search failed with status \(http.statusCode)"
            ])
        }
        struct SearchResponse: Codable {
            let items: [Repo]
        }
        let res = try decoder.decode(SearchResponse.self, from: data)
        return res.items
    }
    
    func latestRelease(owner: String, repo: String) async throws -> Release? {
        let url = URL(string: "https://api.github.com/repos/\(owner)/\(repo)/releases/latest")!
        let (data, http) = try await request(url)
        if http.statusCode == 404 { return nil } // no releases
        guard http.statusCode == 200 else {
            throw NSError(domain: "GitHub", code: http.statusCode, userInfo: [
                NSLocalizedDescriptionKey: "Latest release failed with status \(http.statusCode)"
            ])
        }
        return try decoder.decode(Release.self, from: data)
    }
}

// MARK: - Filtering Helpers

extension Asset {
    var isMacInstallerLike: Bool {
        let lower = name.lowercased()
        let isDMG = lower.hasSuffix(".dmg")
        let isPKG = lower.hasSuffix(".pkg")
        let isZip = lower.hasSuffix(".zip") && (lower.contains("mac") || lower.contains("macos"))
        return isDMG || isPKG || isZip
    }
    
    func matches(arch: CPUArch) -> Bool {
        let lower = name.lowercased()
        switch arch {
        case .arm64:
            // common markers
            return lower.contains("arm64") ||
                   lower.contains("aarch64") ||
                   (lower.contains("apple") && lower.contains("silicon")) ||
                   lower.contains("m1") || lower.contains("m2") || lower.contains("m3") ||
                   // some projects ship universal only
                   lower.contains("universal") ||
                   (!lower.contains("x86") && !lower.contains("intel"))
        case .x86_64:
            return lower.contains("x86_64") ||
                   lower.contains("x64") ||
                   lower.contains("intel") ||
                   lower.contains("amd64") ||
                   lower.contains("universal") ||
                   (!lower.contains("arm") && !lower.contains("silicon"))
        }
    }
}

func filterMacAssets(_ assets: [Asset], arch: CPUArch) -> [Asset] {
    let lowercasedAssets = assets.map { ($0, $0.name.lowercased()) }

    // Step 1: 架构匹配 (arm64 / x86_64)
    let archMatches = lowercasedAssets.filter { $0.0.matches(arch: arch) }
    if !archMatches.isEmpty {
        return archMatches.map { $0.0 }
    }

    // Step 2: 通用 universal
    let universal = lowercasedAssets.filter { $0.1.contains("universal") }
    if !universal.isEmpty {
        return universal.map { $0.0 }
    }

    // Step 3: mac 关键词 (mac, macos, osx, os x)
    let macKeywords = ["macos", "mac", "osx", "os x"]
    let macLike = lowercasedAssets.filter { pair in
        macKeywords.contains { pair.1.contains($0) }
    }
    if !macLike.isEmpty {
        return macLike.map { $0.0 }
    }

    // Step 4: 文件类型 dmg/pkg/app/含关键字的压缩包
    let installerLike = lowercasedAssets.filter { pair in
        pair.1.hasSuffix(".dmg") ||
        pair.1.hasSuffix(".pkg") ||
        pair.1.hasSuffix(".app") ||
        (pair.1.hasSuffix(".zip") &&
         (pair.1.contains(".dmg") || pair.1.contains(".pkg") || pair.1.contains(".app")))
    }
    if !installerLike.isEmpty {
        return installerLike.map { $0.0 }
    }

    // Step 5: 实在没有 → 全部
    return assets
}

// MARK: - App State

// MARK: - New AppState

// MARK: - New AppState

@MainActor
final class AppState: ObservableObject {
    @Published var items: [AppItem] = []
    @Published var isScanning: Bool = false
    @Published var progressText: String = "准备扫描…"
    @Published var errorMessage: String?
    @Published var query: String
    @Published var isManualSearch: Bool = false
    @Published var favoriteIDs: Set<Int> = []

    private let settings: SettingsStore

    /// Whether to auto-start scan on appear (first launch only)
    let shouldAutoStartScanOnAppear: Bool

    private static let favoritesKey = "favorites"

    init(settings: SettingsStore) {
        self.settings = settings
        self.query = settings.recommendedQuery
        // Only auto start scanning on first launch
        self.shouldAutoStartScanOnAppear = !AppState.hasLaunchedBefore
        // Try to load cached items, else empty
        if let cached = Self.loadCachedRecommendedItems() {
            self.items = cached
        } else {
            self.items = []
        }
        loadFavorites()
        if !AppState.hasLaunchedBefore {
            AppState.setLaunched()
        }
    }

    func toggleFavorite(_ item: AppItem) {
        if favoriteIDs.contains(item.id) {
            favoriteIDs.remove(item.id)
        } else {
            favoriteIDs.insert(item.id)
        }
        saveFavorites()
    }

    func favoriteItems(from allItems: [AppItem]) -> [AppItem] {
        allItems.filter { favoriteIDs.contains($0.id) }
    }

    private func saveFavorites() {
        let ids = Array(favoriteIDs)
        UserDefaults.standard.set(ids, forKey: Self.favoritesKey)
    }

    private func loadFavorites() {
        if let ids = UserDefaults.standard.array(forKey: Self.favoritesKey) as? [Int] {
            favoriteIDs = Set(ids)
        }
    }

    // MARK: - Fuzzy Search Helpers
    private static let stopwords: Set<String> = [
        "the", "a", "an", "and", "or", "of", "for", "with", "on", "in", "to", "by", "from",
        "is", "are", "was", "were", "be", "this", "that", "it", "as", "at", "but", "not", "no"
    ]

    private func preprocessKeywords(_ input: String) -> [String] {
        // Lowercase, split by whitespace, remove stopwords, deduplicate, min 2 chars
        let words = input
            .lowercased()
            .components(separatedBy: .whitespacesAndNewlines)
            .map { $0.trimmingCharacters(in: .punctuationCharacters) }
            .filter { !$0.isEmpty && $0.count > 1 && !Self.stopwords.contains($0) }
        return Array(Set(words))
    }

    private func buildQuery(from input: String) -> String {
        let trimmed = input.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty { return settings.recommendedQuery }
        // 如果用户输入了高级语法（含冒号），按原样查询
        if trimmed.contains(":") { return trimmed }
        // 通用搜索：从名称、描述、README 中查找，排除归档仓库
        // Support multiple keywords, join with space (which is OR in GitHub API)
        let keywords = preprocessKeywords(trimmed)
        if keywords.isEmpty { return "in:name,description,readme archived:false" }
        let joined = keywords.joined(separator: " ")
        return "\(joined) in:name,description,readme archived:false"
    }

    func updateQueryAndScan(for input: String) {
        isManualSearch = true
        self.query = buildQuery(from: input)
        startScan(originalInput: input)
        isManualSearch = false
    }

    /// 流式抓取并逐步更新列表，支持高级模糊搜索和README分析
    func startScan(originalInput: String? = nil) {
        // removed: guard isManualSearch else { return }
        guard !isScanning else { return }
        isScanning = true
        errorMessage = nil
        progressText = "准备扫描…"
        items = []

        // README缓存（内存+磁盘）
        actor ReadmeCache {
            private var memCache: [String: String] = [:]
            private let diskURL: URL
            init() {
                let dir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
                diskURL = dir.appendingPathComponent("Macoria-ReadmeCache")
                try? FileManager.default.createDirectory(at: diskURL, withIntermediateDirectories: true)
            }
            func get(_ key: String) -> String? {
                if let v = memCache[key] { return v }
                let file = diskURL.appendingPathComponent(key)
                if let data = try? Data(contentsOf: file), let str = String(data: data, encoding: .utf8) {
                    memCache[key] = str
                    return str
                }
                return nil
            }
            func set(_ key: String, value: String) {
                memCache[key] = value
                let file = diskURL.appendingPathComponent(key)
                try? value.data(using: .utf8)?.write(to: file)
            }
        }
        let readmeCache = ReadmeCache()

        Task {
            let currentQuery = self.query
            let currentSettings = self.settings
            let client = GitHubClient(token: currentSettings.githubToken)
            let inputString = originalInput ?? currentQuery
            let keywords = preprocessKeywords(inputString)
            var collected: [(AppItem, Double)] = []
            var seen = Set<Int>()
            let totalPages = max(1, currentSettings.pageLimit)
            // Snapshot values from settings on MainActor to avoid cross-actor access
            let allowPre = await MainActor.run { currentSettings.prereleaseAllowed }
            let token = await MainActor.run { currentSettings.githubToken }

            // Fuzzy matching helpers
            func fuzzyScore(_ haystack: String, _ keyword: String) -> Double {
                let hay = haystack.lowercased()
                let key = keyword.lowercased()
                if hay.contains(key) { return 1.0 }
                // Levenshtein distance
                func levenshtein(_ a: String, _ b: String) -> Int {
                    let a = Array(a), b = Array(b)
                    var dp = Array(0...b.count)
                    for (i, ca) in a.enumerated() {
                        var prev = dp[0]
                        dp[0] = i + 1
                        for (j, cb) in b.enumerated() {
                            let cost = ca == cb ? 0 : 1
                            let cur = min(dp[j+1] + 1, dp[j] + 1, prev + cost)
                            prev = dp[j+1]
                            dp[j+1] = cur
                        }
                    }
                    return dp[b.count]
                }
                // Accept small typos: e.g., edit distance <= 1 for short words, <=2 for longer
                let dist = levenshtein(hay, key)
                if key.count <= 4 && dist <= 1 { return 0.7 }
                if key.count > 4 && dist <= 2 { return 0.6 }
                // Partial match
                if hay.contains(String(key.prefix(3))) { return 0.5 }
                return 0.0
            }

            func matchScore(repo: Repo, readme: String?, keywords: [String]) -> Double {
                var score = 0.0
                for key in keywords {
                    // Name: weight 3
                    let s1 = fuzzyScore(repo.name, key) * 3
                    // Description: weight 2
                    let s2 = fuzzyScore(repo.description ?? "", key) * 2
                    // README: weight 1
                    let s3 = fuzzyScore(readme ?? "", key) * 1
                    score += max(s1, s2, s3)
                }
                return score
            }

            func finalScore(repo: Repo, matchScore: Double, release: Release) -> Double {
                // Combine: matchScore (main), stars (log), recency (last updated)
                let stars = Double(repo.stargazersCount)
                let starScore = log10(stars + 1) // 0-4
                let updated = release.publishedAt ?? Date.distantPast
                let days = -updated.timeIntervalSinceNow / 86400
                let recency = max(0, 1 - (days / 365)) // within 1 year: 1.0, older: less
                // Weighted sum
                return matchScore * 3 + starScore * 1.5 + recency
            }

            // Helper function to fetch README via GitHub API
            func fetchReadme(owner: String, repo: String, token: String?) async -> String? {
                let url = URL(string: "https://api.github.com/repos/\(owner)/\(repo)/readme")!
                var req = URLRequest(url: url)
                req.setValue("application/vnd.github.VERSION.raw", forHTTPHeaderField: "Accept")
                req.setValue("Macoria/1.0", forHTTPHeaderField: "User-Agent")
                if let token = token, !token.isEmpty {
                    req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
                }
                do {
                    let (data, resp) = try await URLSession.shared.data(for: req)
                    guard let http = resp as? HTTPURLResponse, http.statusCode == 200 else {
                        return nil
                    }
                    return String(data: data, encoding: .utf8)
                } catch {
                    return nil
                }
            }

            for page in 1...totalPages {
                progressText = "检索仓库（第 \(page)/\(totalPages) 页）…"
                do {
                    let repos = try await client.searchRepositories(query: currentQuery, page: page)
                    let count = repos.count
                    var processed = 0

                    try await withThrowingTaskGroup(of: (AppItem, Double)?.self) { group in
                        for repo in repos {
                            group.addTask {
                                guard let rel = try? await client.latestRelease(owner: repo.owner.login, repo: repo.name) else { return nil }
                                if !allowPre && (rel.prerelease || rel.draft) { return nil }
                                let filtered = filterMacAssets(rel.assets, arch: CPUArch.current)
                                let chosen = filtered.isEmpty ? rel.assets : filtered
                                guard !chosen.isEmpty else { return nil }

                                // README fetch & cache (via GitHub API)
                                let readmeKey = "\(repo.owner.login)__\(repo.name).md"
                                var readmeContent: String? = await readmeCache.get(readmeKey)
                                if readmeContent == nil {
                                    if let str = await fetchReadme(owner: repo.owner.login, repo: repo.name, token: token) {
                                        readmeContent = str
                                        await readmeCache.set(readmeKey, value: str)
                                    }
                                }

                                let mscore = matchScore(repo: repo, readme: readmeContent, keywords: keywords)
                                if mscore < 0.2 { return nil } // skip low match
                                let item = AppItem(id: repo.id, repo: repo, release: rel, assets: chosen, readme: readmeContent)
                                let fscore = finalScore(repo: repo, matchScore: mscore, release: rel)
                                return (item, fscore)
                            }
                        }
                        for try await result in group {
                            processed += 1
                            if let (item, score) = result, !seen.contains(item.id) {
                                seen.insert(item.id)
                                collected.append((item, score))
                                // 实时排序（final score降序）并刷新 UI
                                self.items = collected
                                    .sorted { $0.1 > $1.1 }
                                    .map { $0.0 }
                            }
                            self.progressText = "处理发布（\(processed)/\(count)）…"
                        }
                    }
                } catch {
                    self.errorMessage = error.localizedDescription
                }
            }
            self.isScanning = false
            self.progressText = "已发现 \(self.items.count) 个可下载的 macOS 应用"
            // Save to cached recommended items
            Self.saveCachedRecommendedItems(self.items)
        }
    }

    // MARK: - Persistence for recommended items (cache)
    private static let cachedRecommendedItemsKey = "cachedRecommendedItems"
    private static let launchedKey = "hasLaunchedBefore"

    static var hasLaunchedBefore: Bool {
        UserDefaults.standard.bool(forKey: launchedKey)
    }
    static func setLaunched() {
        UserDefaults.standard.set(true, forKey: launchedKey)
    }

    static func saveCachedRecommendedItems(_ items: [AppItem]) {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        if let data = try? encoder.encode(items) {
            UserDefaults.standard.set(data, forKey: cachedRecommendedItemsKey)
        }
    }

    static func loadCachedRecommendedItems() -> [AppItem]? {
        guard let data = UserDefaults.standard.data(forKey: cachedRecommendedItemsKey) else { return nil }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try? decoder.decode([AppItem].self, from: data)
    }
}

// MARK: - Download Manager

final class DownloadManager: NSObject, ObservableObject, URLSessionDownloadDelegate {
    static let shared = DownloadManager()

    struct DownloadTaskInfo: Identifiable {
        let id = UUID()
        let url: URL
        var progress: Double
        var totalBytesWritten: Int64
        var totalBytesExpected: Int64
        var speed: Double
        var estimatedRemaining: TimeInterval?
        var destination: URL?
        var isCompleted: Bool
        var isPaused: Bool
        var resumeData: Data?
        var task: URLSessionDownloadTask?
    }

    @Published var tasks: [URL: DownloadTaskInfo] = [:]

    private lazy var session: URLSession = {
        let config = URLSessionConfiguration.default
        return URLSession(configuration: config, delegate: self, delegateQueue: nil)
    }()

    func download(_ url: URL) {
        // Allow re-download: always start a new download, even if already exists.
        let task = session.downloadTask(with: url)
        tasks[url] = DownloadTaskInfo(url: url, progress: 0.0, totalBytesWritten: 0, totalBytesExpected: 0, speed: 0, estimatedRemaining: nil, destination: nil, isCompleted: false, isPaused: false, resumeData: nil, task: task)
        task.resume()
    }

    func pause(_ url: URL) {
        guard var info = tasks[url], let task = info.task else { return }
        task.cancel { resumeData in
            info.resumeData = resumeData
            info.isPaused = true
            info.task = nil
            self.tasks[url] = info
        }
    }

    func resume(_ url: URL) {
        guard var info = tasks[url], info.isPaused, let data = info.resumeData else { return }
        let task = session.downloadTask(withResumeData: data)
        info.task = task
        info.isPaused = false
        info.resumeData = nil
        tasks[url] = info
        task.resume()
    }

    func cancel(_ url: URL) {
        guard let info = tasks[url] else { return }
        info.task?.cancel()
        tasks[url] = nil
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                    didWriteData bytesWritten: Int64,
                    totalBytesWritten: Int64,
                    totalBytesExpectedToWrite: Int64) {
        guard let url = downloadTask.originalRequest?.url else { return }
        
        DispatchQueue.main.async {
            var info = self.tasks[url] ?? DownloadTaskInfo(url: url, progress: 0.0, totalBytesWritten: 0, totalBytesExpected: 0, speed: 0, estimatedRemaining: nil, destination: nil, isCompleted: false, isPaused: false, resumeData: nil, task: downloadTask)
            info.totalBytesWritten = totalBytesWritten
            info.totalBytesExpected = totalBytesExpectedToWrite
            info.progress = totalBytesExpectedToWrite > 0 ? Double(totalBytesWritten) / Double(totalBytesExpectedToWrite) : 0

            if let startDate = downloadTask.taskDescription.flatMap({ TimeInterval($0) }) {
                let elapsed = Date().timeIntervalSince1970 - startDate
                if elapsed > 0 {
                    info.speed = Double(totalBytesWritten) / elapsed
                    if info.speed > 0 {
                        let remainingBytes = Double(totalBytesExpectedToWrite - totalBytesWritten)
                        info.estimatedRemaining = remainingBytes / info.speed
                    }
                }
            } else {
                downloadTask.taskDescription = "\(Date().timeIntervalSince1970)"
            }

            self.tasks[url] = info
        }
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                    didFinishDownloadingTo location: URL) {
        guard let url = downloadTask.originalRequest?.url else { return }
        let downloads = FileManager.default.urls(for: .downloadsDirectory, in: .userDomainMask).first!
        var dest = downloads.appendingPathComponent(url.lastPathComponent)
        let fileManager = FileManager.default
        // If file exists, add (1), (2), etc.
        let fileExt = dest.pathExtension
        let baseName = dest.deletingPathExtension().lastPathComponent
        var candidate = dest
        var counter = 1
        while fileManager.fileExists(atPath: candidate.path) {
            let newName = fileExt.isEmpty ? "\(baseName) (\(counter))" : "\(baseName) (\(counter)).\(fileExt)"
            candidate = downloads.appendingPathComponent(newName)
            counter += 1
        }
        dest = candidate
        try? fileManager.moveItem(at: location, to: dest)

        DispatchQueue.main.async {
            var info = self.tasks[url]
            info?.progress = 1.0
            info?.isCompleted = true
            info?.destination = dest
            self.tasks[url] = info
        }
    }
}

// MARK: - UI Components

struct AppRow: View {
    let item: AppItem
    
    var body: some View {
        HStack(spacing: 12) {
            AsyncImage(url: item.repo.owner.avatarURL) { phase in
                switch phase {
                case .success(let image):
                    image.resizable().aspectRatio(contentMode: .fill)
                default:
                    Color.secondary.opacity(0.2)
                }
            }
            .frame(width: 44, height: 44)
            .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
            
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text(item.repo.name).font(.headline)
                    Spacer()
                    Label("\(item.repo.stargazersCount)", systemImage: "star.fill")
                        .labelStyle(.iconOnly)
                        .foregroundStyle(.yellow)
                    Text("\(item.repo.stargazersCount)")
                        .foregroundStyle(.secondary)
                        .font(.subheadline)
                }
                Text(item.repo.description ?? "无描述")
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
            }
        }
        .padding(.vertical, 4)
    }
}

struct DetailView: View {
    let item: AppItem
    @Environment(\.openURL) private var openURL
    @EnvironmentObject private var state: AppState

    @State private var isReadmeExpanded = false
    @State private var isAssetsExpanded = false
    @ObservedObject private var downloadManager = DownloadManager.shared

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                HStack(spacing: 12) {
                    AsyncImage(url: item.repo.owner.avatarURL) { phase in
                        switch phase {
                        case .success(let image):
                            image.resizable().aspectRatio(contentMode: .fill)
                        default:
                            Color.secondary.opacity(0.2)
                        }
                    }
                    .frame(width: 72, height: 72)
                    .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))

                    VStack(alignment: .leading) {
                        Text(item.repo.fullName).font(.title3).bold()
                        Text(item.repo.description ?? "无描述").foregroundStyle(.secondary)
                    }
                    Spacer()
                }

                GroupBox {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("最新版本：\(item.release.tagName)")
                        if let date = item.release.publishedAt {
                            Text("发布日期：\(date.formatted(date: .abbreviated, time: .omitted))")
                                .foregroundStyle(.secondary)
                        }
                        if item.release.prerelease { Text("预发布").foregroundStyle(.orange) }
                    }
                } label: {
                    Label("版本信息", systemImage: "shippingbox.fill")
                }

                // README DisclosureGroup
                DisclosureGroup("README", isExpanded: $isReadmeExpanded) {
                    if let md = item.readme {
                        MarkdownWebView(markdown: md)
                            .frame(minHeight: 200, maxHeight: 400)
                    } else {
                        Text("暂无 README")
                            .foregroundStyle(.secondary)
                            .padding()
                    }
                }

                // Assets DisclosureGroup
                DisclosureGroup("可用安装包（\(CPUArch.current.rawValue)）", isExpanded: $isAssetsExpanded) {
                    VStack(alignment: .leading, spacing: 10) {
                        ForEach(item.assets) { asset in
                            if let info = downloadManager.tasks[asset.browserDownloadURL] {
                                VStack(alignment: .leading, spacing: 4) {
                                    HStack {
                                        Image(systemName: info.isCompleted ? "checkmark.circle.fill" : "arrow.down.circle.fill")
                                        Text(asset.name).font(.subheadline).textSelection(.enabled)
                                        Spacer()
                                        if info.isCompleted {
                                            Text("\(ByteCountFormatter.string(fromByteCount: info.totalBytesWritten, countStyle: .file))")
                                            Button {
                                                if let dest = info.destination {
                                                    NSWorkspace.shared.activateFileViewerSelecting([dest])
                                                }
                                            } label: {
                                                Image(systemName: "magnifyingglass")
                                            }
                                            Button {
                                                downloadManager.download(asset.browserDownloadURL)
                                            } label: {
                                                Image(systemName: "arrow.clockwise.circle")
                                            }
                                        } else {
                                            ProgressView(value: info.progress)
                                                .progressViewStyle(.linear)
                                                .frame(width: 120)
                                            Text("\(Int(info.progress * 100))%").font(.caption).foregroundStyle(.secondary)
                                        }
                                    }
                                    if !info.isCompleted {
                                        HStack {
                                            Text("已下载 \(ByteCountFormatter.string(fromByteCount: info.totalBytesWritten, countStyle: .file)) / \(ByteCountFormatter.string(fromByteCount: info.totalBytesExpected, countStyle: .file))")
                                                .font(.caption2)
                                            if info.speed > 0 {
                                                Text("速度 \(ByteCountFormatter.string(fromByteCount: Int64(info.speed), countStyle: .file))/s")
                                                    .font(.caption2)
                                            }
                                            if let remaining = info.estimatedRemaining {
                                                Text("剩余 \(Int(remaining))s")
                                                    .font(.caption2)
                                            }
                                            Spacer()
                                            if info.isPaused {
                                                Button("继续") { downloadManager.resume(asset.browserDownloadURL) }
                                            } else {
                                                Button("暂停") { downloadManager.pause(asset.browserDownloadURL) }
                                            }
                                            Button(role: .destructive) { downloadManager.cancel(asset.browserDownloadURL) } label: {
                                                Image(systemName: "xmark")
                                            }
                                            Button {
                                                downloadManager.download(asset.browserDownloadURL)
                                            } label: {
                                                Image(systemName: "arrow.clockwise.circle")
                                            }
                                        }
                                    }
                                }
                                .padding(.vertical, 2)
                            } else {
                                HStack {
                                    Image(systemName: "arrow.down.circle.fill")
                                    Text(asset.name).font(.subheadline).textSelection(.enabled)
                                    Spacer()
                                    Button {
                                        downloadManager.download(asset.browserDownloadURL)
                                    } label: {
                                        Label("下载", systemImage: "arrow.down.circle")
                                    }
                                    .buttonStyle(.borderedProminent)
                                }
                            }
                        }
                        if item.assets.isEmpty {
                            Text("该发布未提供可识别的 macOS 安装包。已为你展示全部资产或暂无资产。")
                                .font(.footnote)
                                .foregroundStyle(.secondary)
                        }
                    }
                    .padding(.vertical, 4)
                }

                HStack(spacing: 12) {
                    Button {
                        openURL(item.repo.htmlURL)
                    } label: {
                        Label("打开 GitHub", systemImage: "link")
                    }

                    Button {
                        state.toggleFavorite(item)
                    } label: {
                        Label(
                            state.favoriteIDs.contains(item.id) ? "取消收藏" : "收藏",
                            systemImage: state.favoriteIDs.contains(item.id) ? "star.fill" : "star"
                        )
                    }

                    Spacer()
                }
            }
            .padding()
        }
        .navigationTitle(item.repo.name)
    }
}

// MARK: - Settings View

struct SettingsView: View {
    @ObservedObject var settings: SettingsStore

    // For graphical controls
    @State private var minStars: Int = 100
    @State private var earliestUpdate: Date = {
        // Default to 2020-01-01
        var comps = DateComponents()
        comps.year = 2020
        comps.month = 1
        comps.day = 1
        return Calendar.current.date(from: comps) ?? Date(timeIntervalSince1970: 1577836800)
    }()
    @State private var includeForks: Bool = false
    @State private var prereleaseAllowed: Bool = false

    // For Picker
    enum ForkFilter: String, CaseIterable, Identifiable {
        case include = "包含 Fork 仓库"
        case onlyOriginal = "仅原创"
        var id: String { rawValue }
    }
    @State private var forkFilter: ForkFilter = .onlyOriginal

    // Synchronize with settings.recommendedQuery
    private func updateRecommendedQuery() {
        // Compose query string based on graphical controls
        var query = "topic:macos"
        query += " stars:>\(minStars)"
        let dateStr = ISO8601DateFormatter().string(from: earliestUpdate).prefix(10)
        query += " pushed:>\(dateStr)"
        if forkFilter == .onlyOriginal {
            query += " fork:false"
        }
        if !prereleaseAllowed {
            query += " prerelease:false"
        }
        query += " archived:false"
        settings.recommendedQuery = String(query)
        settings.pageLimit = max(1, minStars / 50)
        settings.prereleaseAllowed = prereleaseAllowed
    }

    private func syncFromQuery() {
        // Parse settings.recommendedQuery to update UI controls if possible
        let q = settings.recommendedQuery
        // Parse stars
        if let starsRange = q.range(of: "stars:>") {
            let after = q[starsRange.upperBound...]
            let numStr = after.split(separator: " ").first ?? ""
            if let stars = Int(numStr) {
                minStars = stars
            }
        }
        // Parse pushed:>
        if let pushedRange = q.range(of: "pushed:>") {
            let after = q[pushedRange.upperBound...]
            let dateStr = after.split(separator: " ").first ?? ""
            let formatter = ISO8601DateFormatter()
            if let date = formatter.date(from: String(dateStr)) {
                earliestUpdate = date
            } else if dateStr.count == 10 {
                // fallback for yyyy-MM-dd
                let df = DateFormatter()
                df.dateFormat = "yyyy-MM-dd"
                if let date = df.date(from: String(dateStr)) {
                    earliestUpdate = date
                }
            }
        }
        // Fork filter
        if q.contains("fork:false") {
            forkFilter = .onlyOriginal
        } else {
            forkFilter = .include
        }
        // Prerelease
        if q.contains("prerelease:false") {
            prereleaseAllowed = false
        } else {
            prereleaseAllowed = settings.prereleaseAllowed
        }
    }

    var body: some View {
        TabView {
            // 账号 Tab
            VStack(alignment: .leading, spacing: 0) {
                Form {
                    Section {
                        VStack(alignment: .leading, spacing: 6) {
                            Text("填写 Personal Access Token 可以提升 API 调用速率限制（只需 repo:read 权限）")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            SecureField("Personal Access Token", text: $settings.githubToken)
                                .textFieldStyle(.roundedBorder)
                                .font(.system(.body, design: .monospaced))
                                .frame(maxWidth: 340)
                        }
                    } header: {
                        Text("GitHub 账号")
                            .font(.headline)
                            .padding(.bottom, 2)
                    }
                }
                .formStyle(.grouped)
                Spacer()
            }
            .padding(.leading, 40)
            .padding(.top, 34)
            .tabItem {
                Label("账号", systemImage: "person.crop.circle")
            }

            // 推荐设置 Tab
            VStack(alignment: .leading, spacing: 0) {
                Form {
                    Section {
                        VStack(alignment: .leading, spacing: 14) {
                            Stepper(value: $minStars, in: 0...5000, step: 50, onEditingChanged: { _ in updateRecommendedQuery() }) {
                                Text("最低 Star 数量：\(minStars)+")
                            }
                            .help("只显示 star 数大于该值的仓库")
                            Toggle("允许预发布 (Pre-release)", isOn: $prereleaseAllowed)
                                .onChange(of: prereleaseAllowed) { _, _ in updateRecommendedQuery() }
                            DatePicker("最早更新时间", selection: $earliestUpdate, displayedComponents: .date)
                                .onChange(of: earliestUpdate) { _, _ in updateRecommendedQuery() }
                                .help("只显示该日期之后有更新的仓库")
                            Picker("仓库类型", selection: $forkFilter) {
                                ForEach(ForkFilter.allCases) { option in
                                    Text(option.rawValue).tag(option)
                                }
                            }
                            .onChange(of: forkFilter) { _, _ in updateRecommendedQuery() }
                            .pickerStyle(.segmented)
                        }
                        .padding(.vertical, 4)
                        .onChange(of: minStars) { _, _ in updateRecommendedQuery() }
                    } header: {
                        Text("推荐设置")
                            .font(.headline)
                            .padding(.bottom, 2)
                    }
                }
                .formStyle(.grouped)
                .onAppear {
                    syncFromQuery()
                }
                Spacer()
            }
            .padding(.leading, 40)
            .padding(.top, 34)
            .tabItem {
                Label("推荐设置", systemImage: "slider.horizontal.3")
            }

            // 关于 Tab
            VStack(alignment: .leading, spacing: 18) {
                Spacer()
                HStack(spacing: 16) {
                    Image(systemName: "sparkles")
                        .resizable()
                        .frame(width: 48, height: 48)
                        .foregroundColor(.accentColor)
                    VStack(alignment: .leading, spacing: 3) {
                        Text("Macoria")
                            .font(.title2).bold()
                        Text("版本 1.0")
                            .font(.headline)
                            .foregroundStyle(.secondary)
                    }
                }
                .padding(.bottom, 10)
                Text("Macoria 是一个 macOS 应用，帮助你发现和下载最新的开源 macOS App。\n\n支持通过 GitHub Token 提升 API 速率，内置推荐与自定义搜索，自动过滤适合当前架构的安装包。")
                    .font(.body)
                    .foregroundStyle(.secondary)
                Spacer()
            }
            .padding(.leading, 40)
            .padding(.trailing, 32)
            .tabItem {
                Label("关于", systemImage: "info.circle")
            }
        }
        .padding(.vertical, 20)
        .frame(width: 600, height: 420)
        .background(Color(NSColor.windowBackgroundColor))
    }
}

// MARK: - Content View

struct ContentView: View {
    @EnvironmentObject private var settings: SettingsStore
    @EnvironmentObject private var state: AppState
    @State private var showSettings = false
    @State private var searchText: String = ""
    @State private var selectedItem: AppItem? = nil
    @State private var debounceTask: Task<Void, Never>? = nil
    @State private var lastSubmittedQuery: String = ""
    @StateObject private var downloadManager = DownloadManager.shared
    @State private var showDownloads = false
    
    var body: some View {
        NavigationSplitView {
            sidebar
        } detail: {
            if let selected = selectedItem {
                DetailView(item: selected)
            } else {
                placeholder
            }
        }
        .toolbar {
            ToolbarItemGroup(placement: .automatic) {
                Button {
                    state.startScan()
                } label: {
                    Label("刷新", systemImage: "arrow.clockwise")
                }
                Button {
                    showSettings = true
                } label: {
                    Label("设置", systemImage: "gearshape")
                }
                Button {
                    showDownloads.toggle()
                } label: {
                    ZStack {
                        Image(systemName: "arrow.down.circle")
                        // Show circular progress if any active download exists
                        if let firstTask = downloadManager.tasks.values.first(where: { !$0.isCompleted }) {
                            ProgressView(value: firstTask.progress)
                                .progressViewStyle(.circular)
                                .scaleEffect(0.6)
                        }
                    }
                }
                .popover(isPresented: $showDownloads) {
                    VStack(alignment: .leading, spacing: 12) {
                        Text("下载中").font(.headline)
                        ForEach(Array(downloadManager.tasks.values)) { info in
                            VStack(alignment: .leading, spacing: 4) {
                                HStack {
                                    Image(systemName: info.isCompleted ? "checkmark.circle.fill" : "arrow.down.circle.fill")
                                        .foregroundStyle(info.isCompleted ? .green : .accentColor)
                                    Text(info.url.lastPathComponent)
                                        .font(.subheadline)
                                        .lineLimit(1)
                                    Spacer()
                                    if info.isCompleted {
                                        Text("\(ByteCountFormatter.string(fromByteCount: info.totalBytesWritten, countStyle: .file))")
                                        Button {
                                            if let dest = info.destination {
                                                NSWorkspace.shared.activateFileViewerSelecting([dest])
                                            }
                                        } label: {
                                            Image(systemName: "magnifyingglass")
                                        }
                                        Button {
                                            downloadManager.download(info.url)
                                        } label: {
                                            Image(systemName: "arrow.clockwise.circle")
                                        }
                                    } else {
                                        ProgressView(value: info.progress)
                                            .progressViewStyle(.linear)
                                            .frame(width: 100)
                                        Text("\(Int(info.progress * 100))%")
                                            .font(.caption)
                                            .foregroundStyle(.secondary)
                                    }
                                }
                                if !info.isCompleted {
                                    HStack(spacing: 8) {
                                        Text("已下载 \(ByteCountFormatter.string(fromByteCount: info.totalBytesWritten, countStyle: .file)) / \(ByteCountFormatter.string(fromByteCount: info.totalBytesExpected, countStyle: .file))")
                                            .font(.caption2)
                                        if info.speed > 0 {
                                            Text("速度 \(ByteCountFormatter.string(fromByteCount: Int64(info.speed), countStyle: .file))/s")
                                                .font(.caption2)
                                        }
                                        if let remaining = info.estimatedRemaining {
                                            Text("剩余 \(Int(remaining))s")
                                                .font(.caption2)
                                        }
                                        Spacer()
                                        if info.isPaused {
                                            Button {
                                                downloadManager.resume(info.url)
                                            } label: {
                                                Image(systemName: "play.fill")
                                            }
                                        } else {
                                            Button {
                                                downloadManager.pause(info.url)
                                            } label: {
                                                Image(systemName: "pause.fill")
                                            }
                                        }
                                        Button(role: .destructive) {
                                            downloadManager.cancel(info.url)
                                        } label: {
                                            Image(systemName: "xmark")
                                        }
                                        Button {
                                            downloadManager.download(info.url)
                                        } label: {
                                            Image(systemName: "arrow.clockwise.circle")
                                        }
                                    }
                                }
                            }
                            .padding(.vertical, 3)
                        }
                        if downloadManager.tasks.isEmpty {
                            Text("暂无下载任务")
                                .font(.footnote)
                                .foregroundStyle(.secondary)
                        }
                    }
                    .padding()
                    .frame(width: 340)
                }
            }
        }
        .sheet(isPresented: $showSettings) {
            SettingsView(settings: settings)
        }
        // Removed .onChange(of: state.items) that auto-selects first item to avoid repeated searches
        .navigationTitle("Macoria")
        // Removed .searchable and .onSubmit
        .onAppear {
            if state.shouldAutoStartScanOnAppear {
                state.startScan()
            }
        }
    }
    
    @ViewBuilder
    private var sidebar: some View {
        VStack(spacing: 0) {
            // Search bar at the top (always visible)
            HStack(spacing: 8) {
                TextField("搜索 GitHub…", text: $searchText)
                    .textFieldStyle(.roundedBorder)
                    .submitLabel(.search)
                    .onSubmit {
                        performSearch()
                    }

                Button("搜索") {
                    performSearch()
                }
            }
            .padding([.top, .horizontal], 12)
            .padding(.bottom, 8)

            if state.isScanning {
                VStack(spacing: 8) {
                    ProgressView()
                    Text(state.progressText).font(.footnote).foregroundStyle(.secondary)
                }
                .padding(12)
                .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
                .padding()
            }
            // The selection binding below only updates selectedItem; it does not trigger a search.
            List(selection: $selectedItem) {
                if !state.favoriteIDs.isEmpty {
                    Section("我的收藏") {
                        ForEach(state.favoriteItems(from: state.items)) { fav in
                            AppRow(item: fav)
                                .tag(fav)
                        }
                    }
                }
                Section("推荐项目") {
                    ForEach(filteredItems) { item in
                        AppRow(item: item)
                            .tag(item)
                    }
                }
            }
            .overlay {
                if !state.isScanning && filteredItems.isEmpty {
                    ContentUnavailableView("没有匹配结果", systemImage: "magnifyingglass", description: Text("试试修改检索条件或点击工具栏的“刷新”。"))
                }
            }
        }
    }
    
    // 现在搜索框触发“全 GitHub 搜索”，这里不再本地二次过滤，直接呈现流式结果
    private var filteredItems: [AppItem] {
        // If searchText is empty, show recommended (all state.items) but exclude favorites
        if searchText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return state.items.filter { !state.favoriteIDs.contains($0.id) }
        } else {
            return state.items
        }
    }
    
    @ViewBuilder
    private var placeholder: some View {
        GeometryReader { proxy in
            ZStack {
                VStack(spacing: 20) {
                    Image(systemName: state.isScanning ? "shippingbox.circle.fill" : "sparkles")
                        .resizable()
                        .scaledToFit()
                        .frame(width: 120, height: 120)
                        .foregroundColor(.accentColor)
                        .padding(.bottom, 8)
                    Text(state.isScanning ? "正在检索应用…" : "欢迎来到Macoria")
                        .font(.title).bold()
                    Text(state.isScanning ? "正在连接 GitHub 获取最新开源 macOS 应用" : "点击上方搜索或刷新按钮，开始探索推荐应用")
                        .font(.title3)
                        .foregroundStyle(.secondary)
                }
                .multilineTextAlignment(.center)
                .padding()
            }
            .frame(width: proxy.size.width, height: proxy.size.height, alignment: .center)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(.thinMaterial)
        }
    }

    // MARK: - Search Helper
    private func performSearch() {
        let trimmed = searchText.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty {
            // 空白搜索 → fallback to recommended query string and scan
            state.isManualSearch = false
            state.query = settings.recommendedQueryString
            state.startScan()
            return
        }
        // 正常搜索
        state.isManualSearch = true
        // If the query is empty after trimming, fallback to recommendedQueryString
        if trimmed.isEmpty {
            state.query = settings.recommendedQueryString
        }
        state.updateQueryAndScan(for: trimmed)
    }
}

// MARK: - Entry

@main
struct MacoriaApp: App {
    @StateObject private var settings = SettingsStore()
    @StateObject private var state: AppState

    init() {
        let settings = SettingsStore()
        _settings = StateObject(wrappedValue: settings)
        _state = StateObject(wrappedValue: AppState(settings: settings))
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(state)
                .environmentObject(settings)
                .frame(minWidth: 900, minHeight: 600)
                .navigationSplitViewColumnWidth(min: 240, ideal: 280, max: 300)
                .onAppear {
                    state.isManualSearch = false
                    state.query = settings.recommendedQueryString
                    state.startScan()
                }
        }
    }
}
