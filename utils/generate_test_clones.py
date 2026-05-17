import os
import random
import shutil
import re

# 50 Harici C++ Algoritma ve Veri Yapısı (Zero Data Contamination)
SEEDS = {
    "fibonacci": "int fib(int n) { if (n <= 1) return n; int a=0, b=1, c; for(int i=2; i<=n; i++) {c=a+b; a=b; b=c;} return b; }",
    "factorial": "long long fact(int n) { long long res=1; for(int i=2; i<=n; i++) res*=i; return res; }",
    "bubble_sort": "void bubbleSort(int arr[], int n) { for(int i=0; i<n-1; i++) for(int j=0; j<n-i-1; j++) if(arr[j]>arr[j+1]) {int t=arr[j]; arr[j]=arr[j+1]; arr[j+1]=t;} }",
    "binary_search": "int binSearch(int arr[], int l, int r, int x) { while(l<=r) {int m=l+(r-l)/2; if(arr[m]==x) return m; if(arr[m]<x) l=m+1; else r=m-1;} return -1; }",
    "gcd": "int gcd(int a, int b) { while(b) {int t=b; b=a%b; a=t;} return a; }",
    "lcm": "int lcm(int a, int b) { return (a*b)/gcd(a,b); }",
    "linear_search": "int linSearch(int arr[], int n, int x) { for(int i=0; i<n; i++) if(arr[i]==x) return i; return -1; }",
    "selection_sort": "void selSort(int arr[], int n) { for(int i=0; i<n-1; i++) {int min_i=i; for(int j=i+1; j<n; j++) if(arr[j]<arr[min_i]) min_i=j; int t=arr[i]; arr[i]=arr[min_i]; arr[min_i]=t;} }",
    "insertion_sort": "void insSort(int arr[], int n) { for(int i=1; i<n; i++) {int key=arr[i]; int j=i-1; while(j>=0 && arr[j]>key) {arr[j+1]=arr[j]; j--;} arr[j+1]=key;} }",
    "is_prime": "bool isPrime(int n) { if(n<=1) return false; for(int i=2; i*i<=n; i++) if(n%i==0) return false; return true; }",
    "power": "long long power(int b, int p) { long long r=1; while(p){ if(p&1) r*=b; b*=b; p>>=1; } return r; }",
    "reverse_string": "void revStr(char* str) { int l=0, r=strlen(str)-1; while(l<r) {char t=str[l]; str[l]=str[r]; str[r]=t; l++; r--;} }",
    "palindrome": "bool isPal(char* str) { int l=0, r=strlen(str)-1; while(l<r) {if(str[l++]!=str[r--]) return false;} return true; }",
    "find_max": "int getMax(int arr[], int n) { int m=arr[0]; for(int i=1; i<n; i++) if(arr[i]>m) m=arr[i]; return m; }",
    "find_min": "int getMin(int arr[], int n) { int m=arr[0]; for(int i=1; i<n; i++) if(arr[i]<m) m=arr[i]; return m; }",
    "array_sum": "int arrSum(int arr[], int n) { int s=0; for(int i=0; i<n; i++) s+=arr[i]; return s; }",
    "count_vowels": "int countVowels(char* s) { int c=0; for(int i=0; s[i]; i++) {char x=tolower(s[i]); if(x=='a'||x=='e'||x=='i'||x=='o'||x=='u') c++;} return c; }",
    "swap": "void swap(int* a, int* b) { int t=*a; *a=*b; *b=t; }",
    "sum_digits": "int sumDigits(int n) { int s=0; while(n) {s+=n%10; n/=10;} return s; }",
    "merge_sort_merge": "void merge(int arr[], int l, int m, int r) { int i=l, j=m+1, k=0; int temp[r-l+1]; while(i<=m && j<=r) {if(arr[i]<=arr[j]) temp[k++]=arr[i++]; else temp[k++]=arr[j++];} while(i<=m) temp[k++]=arr[i++]; while(j<=r) temp[k++]=arr[j++]; for(i=l,k=0; i<=r; i++,k++) arr[i]=temp[k]; }",
    "quick_sort_part": "int partition(int arr[], int l, int h) { int p=arr[h]; int i=l-1; for(int j=l; j<h; j++) if(arr[j]<p) {i++; int t=arr[i]; arr[i]=arr[j]; arr[j]=t;} int t=arr[i+1]; arr[i+1]=arr[h]; arr[h]=t; return i+1; }",
    "heapify": "void heapify(int arr[], int n, int i) { int lg=i, l=2*i+1, r=2*i+2; if(l<n && arr[l]>arr[lg]) lg=l; if(r<n && arr[r]>arr[lg]) lg=r; if(lg!=i) {int t=arr[i]; arr[i]=arr[lg]; arr[lg]=t; heapify(arr,n,lg);} }",
    "counting_sort": "void countSort(int arr[], int n) { int max=arr[0]; for(int i=1; i<n; i++) if(arr[i]>max) max=arr[i]; int count[max+1]; for(int i=0; i<=max; ++i) count[i]=0; for(int i=0; i<n; i++) count[arr[i]]++; int idx=0; for(int i=0; i<=max; i++) while(count[i]--) arr[idx++]=i; }",
    "matrix_mult": "void matMult(int a[10][10], int b[10][10], int res[10][10], int n) { for(int i=0; i<n; i++) for(int j=0; j<n; j++) {res[i][j]=0; for(int k=0; k<n; k++) res[i][j]+=a[i][k]*b[k][j];} }",
    "sieve_eratosthenes": "void sieve(int n) { bool p[n+1]; memset(p,true,sizeof(p)); for(int p=2; p*p<=n; p++) if(p[p]) for(int i=p*p; i<=n; i+=p) p[i]=false; }",
    "str_cmp": "int strCmp(char* s1, char* s2) { while(*s1 && (*s1==*s2)) {s1++; s2++;} return *(const unsigned char*)s1 - *(const unsigned char*)s2; }",
    "str_cat": "char* strCat(char* dest, const char* src) { char* rd=dest; while(*dest) dest++; while(*dest++=*src++); return rd; }",
    "str_cpy": "char* strCpy(char* d, const char* s) { char* saved=d; while(*s) *d++=*s++; *d=0; return saved; }",
    "bfs_graph": "void bfs(int s, int adj[][100], int v) { bool vis[v]={false}; int q[v], f=0, r=0; q[r++]=s; vis[s]=true; while(f<r) {int curr=q[f++]; for(int i=0; i<v; i++) if(adj[curr][i] && !vis[i]) {vis[i]=true; q[r++]=i;}} }",
    "dfs_graph": "void dfsUtil(int u, int adj[][100], int v, bool vis[]) { vis[u]=true; for(int i=0; i<v; i++) if(adj[u][i] && !vis[i]) dfsUtil(i,adj,v,vis); }",
    "djikstra": "void dijkstra(int graph[100][100], int src, int V) { int dist[V]; bool sptSet[V]; for(int i=0; i<V; i++) dist[i]=INT_MAX, sptSet[i]=false; dist[src]=0; for(int c=0; c<V-1; c++) {int u=-1; for(int i=0; i<V; i++) if(!sptSet[i] && (u==-1 || dist[i]<dist[u])) u=i; sptSet[u]=true; for(int v=0; v<V; v++) if(!sptSet[v] && graph[u][v] && dist[u]!=INT_MAX && dist[u]+graph[u][v]<dist[v]) dist[v]=dist[u]+graph[u][v];} }",
    "floyd_warshall": "void floyd(int graph[][100], int V) { int dist[V][V]; for(int i=0; i<V; i++) for(int j=0; j<V; j++) dist[i][j]=graph[i][j]; for(int k=0; k<V; k++) for(int i=0; i<V; i++) for(int j=0; j<V; j++) if(dist[i][k]+dist[k][j]<dist[i][j]) dist[i][j]=dist[i][k]+dist[k][j]; }",
    "kruskal_mst": "int find(int i, int parent[]) { while(parent[i]) i=parent[i]; return i; } void uni(int i, int j, int parent[]) { if(i!=j) {parent[j]=i; return;} }",
    "knapsack_dp": "int knapSack(int W, int wt[], int val[], int n) { int dp[n+1][W+1]; for(int i=0; i<=n; i++) {for(int w=0; w<=W; w++) {if(i==0||w==0) dp[i][w]=0; else if(wt[i-1]<=w) dp[i][w]=max(val[i-1]+dp[i-1][w-wt[i-1]], dp[i-1][w]); else dp[i][w]=dp[i-1][w];}} return dp[n][W]; }",
    "lcs_dp": "int lcs(char* X, char* Y, int m, int n) { int L[m+1][n+1]; for(int i=0; i<=m; i++) {for(int j=0; j<=n; j++) {if(i==0||j==0) L[i][j]=0; else if(X[i-1]==Y[j-1]) L[i][j]=L[i-1][j-1]+1; else L[i][j]=max(L[i-1][j],L[i][j-1]);}} return L[m][n]; }",
    "lis_dp": "int lis(int arr[], int n) { int lis[n]; for(int i=0; i<n; i++) lis[i]=1; for(int i=1; i<n; i++) for(int j=0; j<i; j++) if(arr[i]>arr[j] && lis[i]<lis[j]+1) lis[i]=lis[j]+1; int m=0; for(int i=0; i<n; i++) if(m<lis[i]) m=lis[i]; return m; }",
    "edit_distance": "int editDist(char* s1, char* s2, int m, int n) { int dp[m+1][n+1]; for(int i=0; i<=m; i++) {for(int j=0; j<=n; j++) {if(i==0) dp[i][j]=j; else if(j==0) dp[i][j]=i; else if(s1[i-1]==s2[j-1]) dp[i][j]=dp[i-1][j-1]; else dp[i][j]=1+min(min(dp[i][j-1],dp[i-1][j]),dp[i-1][j-1]);}} return dp[m][n]; }",
    "subset_sum": "bool isSubsetSum(int set[], int n, int sum) { bool sub[n+1][sum+1]; for(int i=0; i<=n; i++) sub[i][0]=true; for(int i=1; i<=sum; i++) sub[0][i]=false; for(int i=1; i<=n; i++) for(int j=1; j<=sum; j++) {if(j<set[i-1]) sub[i][j]=sub[i-1][j]; else sub[i][j]=sub[i-1][j] || sub[i-1][j-set[i-1]];} return sub[n][sum]; }",
    "coin_change": "int count(int S[], int m, int n) { int table[n+1]; memset(table, 0, sizeof(table)); table[0]=1; for(int i=0; i<m; i++) for(int j=S[i]; j<=n; j++) table[j]+=table[j-S[i]]; return table[n]; }",
    "kadane_max_subarray": "int maxSubArraySum(int a[], int size) { int max_so_far=INT_MIN, max_ending_here=0; for(int i=0; i<size; i++) {max_ending_here=max_ending_here+a[i]; if(max_so_far<max_ending_here) max_so_far=max_ending_here; if(max_ending_here<0) max_ending_here=0;} return max_so_far; }",
    "kmp_search": "void computeLPS(char* pat, int M, int* lps) { int len=0; lps[0]=0; int i=1; while(i<M) {if(pat[i]==pat[len]) {len++; lps[i]=len; i++;} else {if(len!=0) len=lps[len-1]; else {lps[i]=0; i++;}}} }",
    "rabin_karp": "void search(char pat[], char txt[], int q) { int M=strlen(pat), N=strlen(txt); int p=0, t=0, h=1, d=256; for(int i=0; i<M-1; i++) h=(h*d)%q; for(int i=0; i<M; i++) {p=(d*p+pat[i])%q; t=(d*t+txt[i])%q;} for(int i=0; i<=N-M; i++) {if(p==t) {bool m=true; for(int j=0; j<M; j++) if(txt[i+j]!=pat[j]) m=false; if(m) printf(\"found\");} if(i<N-M) {t=(d*(t-txt[i]*h)+txt[i+M])%q; if(t<0) t+=q;}} }",
    "trie_insert": "void insert(struct TrieNode *root, const char *key) { struct TrieNode *pCrawl = root; for (int i = 0; i < strlen(key); i++) { int index = key[i] - 'a'; if (!pCrawl->children[index]) pCrawl->children[index] = getNode(); pCrawl = pCrawl->children[index]; } pCrawl->isEndOfWord = true; }",
    "bst_insert": "struct node* insert(struct node* node, int key) { if(node==NULL) return newNode(key); if(key<node->key) node->left=insert(node->left,key); else if(key>node->key) node->right=insert(node->right,key); return node; }",
    "bst_search": "struct node* search(struct node* root, int key) { if(root==NULL || root->key==key) return root; if(root->key < key) return search(root->right, key); return search(root->left, key); }",
    "avl_rotations": "struct Node* rightRotate(struct Node *y) { struct Node *x = y->left; struct Node *T2 = x->right; x->right = y; y->left = T2; y->height = max(height(y->left), height(y->right))+1; x->height = max(height(x->left), height(x->right))+1; return x; }",
    "segment_tree_build": "void build(int node, int start, int end) { if(start==end) {tree[node]=A[start]; return;} int mid=(start+end)/2; build(2*node,start,mid); build(2*node+1,mid+1,end); tree[node]=tree[2*node]+tree[2*node+1]; }",
    "fenwick_tree_add": "void update(int i, int delta) { while(i<=n) {BIT[i]+=delta; i += i & (-i);} }",
    "topological_sort": "void topSortUtil(int v, bool vis[], stack<int>& Stack) { vis[v]=true; for(auto i=adj[v].begin(); i!=adj[v].end(); ++i) if(!vis[*i]) topSortUtil(*i, vis, Stack); Stack.push(v); }",
    "bellman_ford": "void BellmanFord(struct Graph* graph, int src) { int V=graph->V, E=graph->E; int dist[V]; for(int i=0; i<V; i++) dist[i]=INT_MAX; dist[src]=0; for(int i=1; i<=V-1; i++) for(int j=0; j<E; j++) {int u=graph->edge[j].src; int v=graph->edge[j].dest; int weight=graph->edge[j].weight; if(dist[u]!=INT_MAX && dist[u]+weight<dist[v]) dist[v]=dist[u]+weight;} }"
}

TYPE_4_PAIRS = [
    # Iterative vs Recursive Fib
    ("int fib(int n) { int a=0, b=1; while(n--) {int t=a+b; a=b; b=t;} return a; }",
     "int fib(int n) { if(n<=1) return n; return fib(n-1)+fib(n-2); }"),
    # Merge Sort Top-Down vs Bottom-Up
    ("void mergeSort(int arr[], int l, int r) { if(l<r) {int m=l+(r-l)/2; mergeSort(arr,l,m); mergeSort(arr,m+1,r); merge(arr,l,m,r);} }",
     "void mergeSort(int arr[], int n) { for(int curr_size=1; curr_size<=n-1; curr_size=2*curr_size) { for(int left_start=0; left_start<n-1; left_start+=2*curr_size) { int mid = min(left_start + curr_size - 1, n-1); int right_end = min(left_start + 2*curr_size - 1, n-1); merge(arr, left_start, mid, right_end); } } }"),
    # DFS Recursive vs Stack
    ("void dfs(int u) { vis[u]=1; for(int v:adj[u]) if(!vis[v]) dfs(v); }",
     "void dfs(int start) { stack<int> s; s.push(start); while(!s.empty()) { int u=s.top(); s.pop(); if(!vis[u]) { vis[u]=1; for(int v:adj[u]) s.push(v); } } }"),
    # GCD
    ("int gcd(int a, int b) { while(b) {int t=b; b=a%b; a=t;} return a; }",
     "int gcd(int a, int b) { return b==0 ? a : gcd(b, a%b); }"),
    # Array Sum
    ("int sum(int arr[], int n) { int s=0; for(int i=0; i<n; i++) s+=arr[i]; return s; }",
     "int sum(int arr[], int n) { if(n==0) return 0; return arr[n-1] + sum(arr, n-1); }"),
    # Palindrome Check
    ("bool isPal(char* s) { int len=strlen(s); for(int i=0; i<len/2; i++) if(s[i]!=s[len-1-i]) return false; return true; }",
     "bool isPal(char* s, int l, int r) { if(l>=r) return true; if(s[l]!=s[r]) return false; return isPal(s, l+1, r-1); }"),
    # Prime Check
    ("bool isPrime(int n) { if(n<2) return false; for(int i=2; i*i<=n; i++) if(n%i==0) return false; return true; }",
     "bool isPrime(int n) { if(n<2) return false; if(n==2||n==3) return true; if(n%2==0||n%3==0) return false; for(int i=5; i*i<=n; i+=6) if(n%i==0 || n%(i+2)==0) return false; return true; }"),
    # Strlen
    ("int getLen(char* str) { int l=0; while(str[l]) l++; return l; }",
     "int getLen(char* str) { char* p = str; while(*p) p++; return p - str; }"),
    # Swap
    ("void swap(int* a, int* b) { int t=*a; *a=*b; *b=t; }",
     "void swap(int* a, int* b) { if(a!=b) { *a=*a^*b; *b=*a^*b; *a=*a^*b; } }"),
    # Factorial
    ("long fact(int n) { long r=1; while(n>1) r*=n--; return r; }",
     "long fact(int n) { return n<=1 ? 1 : n*fact(n-1); }"),
    # Subarray Max
    ("int maxSubArray(int a[], int n) { int m=INT_MIN, end=0; for(int i=0; i<n; i++) {end+=a[i]; if(m<end) m=end; if(end<0) end=0;} return m; }",
     "int maxSubArray(int a[], int l, int h) { if(l==h) return a[l]; int m=(l+h)/2; return max(max(maxSubArray(a,l,m), maxSubArray(a,m+1,h)), maxCrossing(a,l,m,h)); }"),
    # Count Bits
    ("int countBits(int n) { int c=0; while(n) {c+=n&1; n>>=1;} return c; }",
     "int countBits(int n) { int c=0; while(n) {n&=n-1; c++;} return c; }"),
    # Reverse Array
    ("void rev(int arr[], int n) { for(int i=0; i<n/2; i++) swap(&arr[i], &arr[n-1-i]); }",
     "void rev(int arr[], int l, int r) { if(l>=r) return; swap(&arr[l], &arr[r]); rev(arr, l+1, r-1); }"),
    # Binary Search
    ("int bs(int a[], int n, int x) { int l=0, r=n-1; while(l<=r) {int m=l+(r-l)/2; if(a[m]==x) return m; if(a[m]<x) l=m+1; else r=m-1;} return -1; }",
     "int bs(int a[], int l, int r, int x) { if(r>=l) {int m=l+(r-l)/2; if(a[m]==x) return m; if(a[m]>x) return bs(a,l,m-1,x); return bs(a,m+1,r,x);} return -1; }")
]

KEYWORDS = frozenset({
    "int","float","double","char","void","long","short","unsigned","bool","class","struct","enum",
    "return","if","else","for","while","do","break","continue","switch","case","default",
    "public","private","protected","const","static","inline","sizeof",
    "true","false","NULL","nullptr","std","cout","cin","endl","printf","scanf","string","vector","map","set",
    "min","max","memset","strlen","tolower","toupper","INT_MAX","INT_MIN","max_so_far","max_ending_here"
})

def apply_type1(code):
    lines = code.split('\n')
    new_lines = []
    for line in lines:
        if random.random() < 0.2:
            new_lines.append(f"// {random.choice(['TODO', 'Fix this', 'Check logic', 'Optimization needed'])}")
        new_lines.append((" " * random.randint(0, 4)) + line + (" " * random.randint(0, 4)))
        if random.random() < 0.1:
            new_lines.append("")
    return '\n'.join(new_lines)

def apply_type2(code: str) -> str:
    words = re.findall(r'[a-zA-Z_]\w*', code)
    user_words = set(w for w in words if w not in KEYWORDS)
    
    mapping = {}
    for i, w in enumerate(user_words):
        if len(w) <= 2:
            mapping[w] = f"v_{w}{random.randint(1,9)}"
        else:
            mapping[w] = f"var_{i}_{random.randint(100,999)}"
            
    for old, new in mapping.items():
        code = re.sub(rf'(?<![a-zA-Z_0-9]){old}(?![a-zA-Z_0-9])', new, code)
        
    return code

def apply_type3(code):
    code = apply_type2(code)
    lines = code.split(';')
    new_lines = []
    
    for line in lines:
        r = random.random()
        if r < 0.1 and line.strip() and not "{" in line and not "}" in line and not "return" in line:
            # 10% sime: pass
            continue
        new_lines.append(line)
        if r > 0.85:
            # 15% ekleme
            new_lines.append(f" int d_{random.randint(0,99)} = {random.randint(1,10)}")
            
    res = ';'.join(new_lines)
    
    # format
    return res.replace("{", "{\n").replace("}", "\n}\n").replace(";", ";\n")

def generate_pairs(scenario="original"):
    out_dir = f"test_clones_{scenario}"
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    
    types = ["type1", "type2", "type3", "type4", "negatives"]
    for t in types:
        os.makedirs(os.path.join(out_dir, t))
        
    all_snippets = []
        
    if scenario == "original":
        pos_t123 = 120
        pos_t4 = 110
        neg_count = 120
    elif scenario == "imbalanced":
        pos_t123 = 12
        pos_t4 = 14
        neg_count = 950
    elif scenario == "balanced":
        pos_t123 = 125
        pos_t4 = 125
        neg_count = 500
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
        
    seed_keys = list(SEEDS.keys())
    
    for t_name in ["type1", "type2", "type3"]:
        pair_idx = 1
        while pair_idx <= pos_t123:
            for seed_k in seed_keys:
                if pair_idx > pos_t123:
                    break
                base_code = SEEDS[seed_k]
                if t_name == "type1": clone_code = apply_type1(base_code)
                elif t_name == "type2": clone_code = apply_type2(base_code)
                elif t_name == "type3": clone_code = apply_type3(base_code)
                
                pair_dir = os.path.join(out_dir, t_name, f"pair_{pair_idx:03d}")
                os.makedirs(pair_dir)
                with open(os.path.join(pair_dir, "original.txt"), "w") as f: f.write(base_code)
                with open(os.path.join(pair_dir, "clone.txt"), "w") as f: f.write(clone_code)
                
                all_snippets.append((seed_k, base_code))
                all_snippets.append((seed_k, clone_code))
                pair_idx += 1

    pair_idx = 1
    while pair_idx <= pos_t4:
        for base, alt in TYPE_4_PAIRS:
            if pair_idx > pos_t4:
                break
            c1 = base if random.random() < 0.15 else apply_type1(apply_type2(base))
            c2 = alt  if random.random() < 0.15 else apply_type1(apply_type2(alt))
            
            pair_dir = os.path.join(out_dir, "type4", f"pair_{pair_idx:03d}")
            os.makedirs(pair_dir)
            with open(os.path.join(pair_dir, "original.txt"), "w") as f: f.write(c1)
            with open(os.path.join(pair_dir, "clone.txt"), "w") as f: f.write(c2)
            
            all_snippets.append((f"t4_{pair_idx}", c1))
            all_snippets.append((f"t4_{pair_idx}", c2))
            pair_idx += 1

    print(f"Generating {neg_count} negative pairs for scenario '{scenario}'...")
    pair_idx = 1
    negative_pairs = set()
    max_attempts = neg_count * 500
    attempts = 0
    
    while pair_idx <= neg_count and attempts < max_attempts:
        s1 = random.choice(all_snippets)
        s2 = random.choice(all_snippets)
        
        if s1[0] != s2[0]:
            pair_hash = hash((s1[1], s2[1]))
            if pair_hash not in negative_pairs:
                negative_pairs.add(pair_hash)
                
                pair_dir = os.path.join(out_dir, "negatives", f"pair_{pair_idx:04d}")
                os.makedirs(pair_dir)
                with open(os.path.join(pair_dir, "original.txt"), "w") as f: f.write(s1[1])
                with open(os.path.join(pair_dir, "clone.txt"), "w") as f: f.write(s2[1])
                pair_idx += 1
        attempts += 1

    print(f"Dataset generated successfully in {out_dir}/")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate code clone test pairs")
    parser.add_argument("--scenario", type=str, default="all", choices=["original", "imbalanced", "balanced", "all"])
    args = parser.parse_args()
    
    if args.scenario == "all":
        generate_pairs("original")
        generate_pairs("imbalanced")
        generate_pairs("balanced")
    else:
        generate_pairs(args.scenario)
