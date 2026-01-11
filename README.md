# 💻 Cursor Clone - AI-Powered Code Editor

> Intelligent coding assistant with context-aware completions, works offline

## Features

✅ **AI Code Completion**
- Inline suggestions (like GitHub Copilot)
- Context-aware (analyzes entire codebase)
- Multi-language support (TS, Python, Rust, Go)
- Local LLM (CodeLlama 7B)

✅ **AI Chat Assistant**
- Ask questions about code
- Explain complex functions
- Debug errors with AI
- Refactor code suggestions

✅ **Codebase Understanding**
- Semantic code search
- Symbol indexing
- Dependency graph
- "Go to definition" across files

✅ **Smart Editing**
- Multi-cursor editing
- AST-aware refactoring
- Auto-formatting (Prettier/Black)
- Real-time error detection

## Tech Stack

**Editor:** Monaco Editor (VS Code core)  
**AI Runtime:** ONNX Runtime + WebGPU  
**Language Parsing:** Tree-sitter (fast AST)  
**Embeddings:** MiniLM-L6 (semantic search)  
**LLM:** CodeLlama-7B-4bit or Phi-2  
**LSP:** Custom Language Server Protocol  

## AI Models Used

| Model | Size | Purpose | Platform |
|-------|------|---------|----------|
| CodeLlama-7B-4bit | 1.5GB | Code completion | Desktop |
| StarCoder-1B | 600MB | Code completion | Mobile |
| CodeBERT | 400MB | Code understanding | All |
| MiniLM-L6 | 80MB | Code embeddings | All |
| Phi-2 | 1.5GB | Alternative LLM | Desktop |

## Installation

```bash
npm install

# Download models
npm run download-models

# Development
npm run dev

# Build
npm run build:desktop   # Tauri app
npm run build:mobile    # PWA for mobile
```

## Usage

### 1. Inline Code Completion

As you type, AI suggests completions:

```typescript
// You type:
function calculateTax(

// AI suggests:
amount: number, rate: number): number {
  return amount * rate;
}
```

**How it works:**
- Analyzes cursor position & context
- Finds similar code in codebase (embeddings)
- Runs LLM with context window
- Ranks suggestions by confidence

### 2. AI Chat for Code Help

```
User: "How does this function work?"

AI: This function implements a binary search algorithm. 
    It takes a sorted array and a target value...
    
    Time complexity: O(log n)
    Space complexity: O(1)
```

### 3. Code Refactoring

```
User: "Extract this into a reusable component"

AI: [Generates new file with extracted component]
    [Updates imports automatically]
    [Adds TypeScript types]
```

## Key Algorithms

### 1. Context Building (RAG for Code)

Retrieval-Augmented Generation for code completion.

```typescript
async function buildContext(cursorPosition: Position): Promise<string> {
  // Step 1: Get current file content
  const currentFile = getFileContent(cursorPosition.file);
  
  // Step 2: Extract imports (dependencies)
  const imports = parseImports(currentFile);
  
  // Step 3: Get related files via embeddings
  const embedding = await getEmbedding(currentFile);
  const similarFiles = await semanticSearch(embedding, topK=5);
  
  // Step 4: Build context string
  return `
    Current file: ${currentFile}
    
    Related files:
    ${similarFiles.map(f => f.content).join('\n')}
    
    Cursor position: Line ${cursorPosition.line}
  `;
}
```

**Math behind semantic search:**
```
Cosine Similarity = dot(embedding_A, embedding_B) / (||A|| * ||B||)

Where:
  embedding = MiniLM-L6(code_text) → 384-dimensional vector
  Top-K retrieval: Return K files with highest similarity scores
```

### 2. AST-Based Code Analysis

Uses Tree-sitter for fast parsing.

```typescript
function analyzeCode(code: string, language: string): Analysis {
  const tree = parser.parse(code);
  const rootNode = tree.rootNode;
  
  // Extract symbols (functions, classes, variables)
  const symbols = extractSymbols(rootNode);
  
  // Build dependency graph
  const imports = findImports(rootNode);
  const exports = findExports(rootNode);
  
  // Find unused variables
  const unusedVars = detectUnusedVariables(rootNode);
  
  return { symbols, imports, exports, unusedVars };
}
```

**Symbol Table Structure:**
```typescript
interface Symbol {
  name: string;
  type: 'function' | 'class' | 'variable';
  location: { line: number; column: number };
  scope: 'global' | 'local';
  references: Position[];
}
```

### 3. Intelligent Code Completion Ranking

```typescript
function rankCompletions(suggestions: Suggestion[]): Suggestion[] {
  return suggestions.sort((a, b) => {
    const scoreA = 
      (a.frequency * 0.3) +           // How often used in codebase
      (a.recency * 0.2) +             // Recently used
      (a.contextMatch * 0.4) +        // Semantic similarity
      (a.typeMatch * 0.1);            // Type compatibility
    
    const scoreB = /* same formula */;
    
    return scoreB - scoreA;
  });
}
```

**Frequency calculation:**
```
frequency = usageCount / totalSymbols
recency = 1 / (daysSinceLastUse + 1)
contextMatch = cosineSimilarity(currentContext, suggestionContext)
typeMatch = 1 if types compatible, else 0
```

### 4. Error Detection & Debugging

```typescript
async function detectErrors(code: string): Promise<Error[]> {
  const errors: Error[] = [];
  
  // Step 1: Syntax errors (AST parsing)
  const syntaxErrors = parseSyntax(code);
  errors.push(...syntaxErrors);
  
  // Step 2: Type errors (TypeScript compiler)
  const typeErrors = checkTypes(code);
  errors.push(...typeErrors);
  
  // Step 3: AI-based bug detection
  const aiSuggestions = await runBugDetectionModel(code);
  errors.push(...aiSuggestions);
  
  return errors;
}
```

### 5. Multi-File Refactoring

```typescript
async function refactorExtractFunction(
  selection: CodeRange,
  newFunctionName: string
): Promise<FileEdit[]> {
  // Step 1: Analyze selected code
  const ast = parseCode(selection.code);
  const usedVariables = findUsedVariables(ast);
  
  // Step 2: Generate function signature
  const params = usedVariables.filter(v => !v.declaredInSelection);
  const returnType = inferReturnType(ast);
  
  // Step 3: Create new function
  const newFunction = `
    function ${newFunctionName}(${params.join(', ')}): ${returnType} {
      ${selection.code}
    }
  `;
  
  // Step 4: Replace original code with function call
  const replacement = `${newFunctionName}(${params.map(p => p.name).join(', ')})`;
  
  return [
    { file: selection.file, type: 'insert', content: newFunction },
    { file: selection.file, type: 'replace', range: selection, content: replacement }
  ];
}
```

## Model Integration

### CodeLlama Inference

```typescript
class CodeCompletionEngine {
  private model: ort.InferenceSession;
  private tokenizer: Tokenizer;
  
  async complete(context: string, maxTokens: number = 50): Promise<string> {
    // Tokenize input
    const inputIds = this.tokenizer.encode(context);
    
    // Run inference with KV-cache optimization
    let output = '';
    for (let i = 0; i < maxTokens; i++) {
      const feeds = { input_ids: inputIds };
      const results = await this.model.run(feeds);
      
      // Sample next token (top-p sampling)
      const nextToken = this.sampleTopP(results.logits, p=0.9);
      
      if (this.isStopToken(nextToken)) break;
      
      output += this.tokenizer.decode([nextToken]);
      inputIds.push(nextToken);
    }
    
    return output;
  }
  
  // Top-p (nucleus) sampling
  private sampleTopP(logits: Float32Array, p: number): number {
    const probs = softmax(logits);
    const sorted = probs.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]);
    
    let cumProb = 0;
    for (const [prob, idx] of sorted) {
      cumProb += prob;
      if (cumProb >= p) return idx;
    }
    
    return sorted[0][1];
  }
}
```

### Semantic Code Search

```typescript
async function searchCode(query: string): Promise<CodeSnippet[]> {
  // Get query embedding
  const queryEmbedding = await getEmbedding(query);
  
  // Search in vector database
  const results = await vectorDB.search(queryEmbedding, topK=10);
  
  // Re-rank by relevance
  const reranked = results.sort((a, b) => {
    const scoreA = (a.similarity * 0.6) + (a.recency * 0.2) + (a.popularity * 0.2);
    const scoreB = /* same */;
    return scoreB - scoreA;
  });
  
  return reranked;
}
```

## Storage Schema

### Workspace State
```typescript
interface Workspace {
  id: string;
  rootPath: string;
  openFiles: string[];
  cursorPositions: { [file: string]: Position };
  breakpoints: { [file: string]: number[] };
}
```

### Code Index
```typescript
interface CodeIndex {
  fileHash: string;
  symbols: Symbol[];
  imports: Import[];
  exports: Export[];
  embedding: Float32Array;
  lastIndexed: Date;
}
```

### AI Conversations
```typescript
interface Conversation {
  id: string;
  context: {
    files: string[];
    selection?: CodeRange;
  };
  messages: Message[];
}
```

## Performance Optimizations

1. **Incremental Parsing:** Only re-parse changed portions
2. **KV-Cache:** Reuse previous LLM computations
3. **Lazy Indexing:** Index files on-demand
4. **WebWorkers:** Run AI inference in background
5. **Quantization:** 4-bit models (75% smaller)

## Platform Differences

### Desktop (Tauri)
- Full CodeLlama-7B (better completions)
- Multi-threaded indexing
- Native file system access
- GPU acceleration (Metal/Vulkan)
- Larger context window (8192 tokens)

### Mobile/PWA
- Smaller StarCoder-1B
- Limited to 2048 token context
- Simplified UI (no multi-cursor)
- Touch-optimized editor
- Background sync when online

## API Reference

### Completion API
```typescript
interface CompletionEngine {
  complete(position: Position, context: string): Promise<Suggestion[]>;
  acceptCompletion(suggestion: Suggestion): void;
  rejectCompletion(): void;
}
```

### Chat API
```typescript
interface AIChat {
  ask(question: string, context?: CodeRange): Promise<string>;
  explainCode(range: CodeRange): Promise<Explanation>;
  fixError(error: Error): Promise<Fix>;
  refactor(code: string, instructions: string): Promise<string>;
}
```

### Indexing API
```typescript
interface CodeIndexer {
  indexWorkspace(path: string): Promise<void>;
  searchSymbol(name: string): Promise<Symbol[]>;
  findReferences(symbol: Symbol): Promise<Reference[]>;
  getDefinition(position: Position): Promise<Location>;
}
```

## Keyboard Shortcuts

- `Ctrl+Space`: Trigger completion
- `Ctrl+K`: AI chat
- `Ctrl+Shift+R`: Refactor
- `Ctrl+Shift+F`: Semantic search
- `F12`: Go to definition
- `Alt+F12`: Peek definition

## Roadmap

- [x] Inline code completion
- [x] AI chat assistant
- [x] Semantic code search
- [x] AST-based refactoring
- [ ] Collaborative editing
- [ ] Git integration
- [ ] Extension system
- [ ] Cloud sync (optional)

## License

MIT

## Contributing

See CONTRIBUTING.md for development setup and guidelines.
