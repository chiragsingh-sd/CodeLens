const State = {
    currentScreen:   'landing',
    repoId:          null,
    repoName:        null,
    mode:            'chat',
    pollingInterval: null,
    chatHistory:     [],
    selectedFile:    null,
};


function showScreen(screenName) {
    document.getElementById('screen-landing').classList.add('hidden');
    document.getElementById('screen-indexing').classList.add('hidden');
    document.getElementById('screen-chat').classList.add('hidden');

    document.getElementById(`screen-${screenName}`).classList.remove('hidden');

    State.currentScreen = screenName;
}



const API_BASE = '';

async function apiPost(path, body) {
    const response = await fetch(`${API_BASE}${path}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    });

    const data = await response.json();

    if (!response.ok) {
        throw new Error(data.detail || 'Request failed');
    }

    return data;
}

async function apiGet(path) {
    const response = await fetch(`${API_BASE}${path}`);
    const data = await response.json();

    if (!response.ok) {
        throw new Error(data.detail || 'Request failed');
    }

    return data;
}

async function apiDelete(path) {
    const response = await fetch(`${API_BASE}${path}`, {
        method: 'DELETE',
    });
    const data = await response.json();
    if (!response.ok) {
        throw new Error(data.detail || 'Delete failed');
    }
    return data;
}



function validateGithubUrl(url) {
    const trimmed = url.trim();
    return trimmed.startsWith('https://github.com/') && 
           trimmed.split('/').length >= 5;
}

function showUrlError(message) {
    const errorEl = document.getElementById('url-error');
    errorEl.textContent = message;
    errorEl.classList.remove('hidden');
}

function hideUrlError() {
    document.getElementById('url-error').classList.add('hidden');
}

async function handleAnalyzeClick() {
    const urlInput = document.getElementById('repo-url');
    const url      = urlInput.value.trim();
    const btn      = document.getElementById('analyze-btn');

    if (!url) {
        showUrlError('Please enter a GitHub repository URL.');
        return;
    }
    if (!validateGithubUrl(url)) {
        showUrlError('That doesn\'t look like a GitHub URL. Try: https://github.com/owner/repo');
        return;
    }

    hideUrlError();

    btn.disabled = true;
    btn.textContent = 'Submitting...';

    try {
        const data = await apiPost('/api/repos/', { url });

        State.repoId   = data.repo_id;
        State.repoName = url.replace('https://github.com/', '');

        showScreen('indexing');
        document.getElementById('indexing-repo-url').textContent = url;
        startPolling();

    } catch (error) {
        showUrlError(error.message);
        btn.disabled = false;
        btn.textContent = 'Analyze';
    }
}

async function loadPreviousRepos() {
    try {
        const data = await apiGet('/api/repos/');
        const repos = data.repos.filter(r => r.status === 'ready');

        if (repos.length === 0) return;

        const section = document.getElementById('previous-repos-section');
        const list    = document.getElementById('previous-repos-list');

        section.classList.remove('hidden');
        list.innerHTML = '';

        repos.forEach(repo => {
            const card = document.createElement('div');
            card.className = 'repo-card';
            card.innerHTML = `
                <div>
        <div class="repo-card-name">${repo.owner}/${repo.name}</div>
        <div class="repo-card-meta">${repo.chunk_count} chunks · ${repo.file_count} files</div>
    </div>
    <div style="display:flex; align-items:center; gap:8px;">
        <span class="repo-card-arrow">→</span>
        <button 
            class="delete-repo-btn" 
            data-repo-id="${repo.id}"
            data-repo-name="${repo.owner}/${repo.name}"
            title="Delete this repo"
        >✕</button>
    </div>
            `;
           card.addEventListener('click', (e) => {
    if (e.target.closest('.delete-repo-btn')) return;
    State.repoId   = repo.id;
    State.repoName = `${repo.owner}/${repo.name}`;
    openChatScreen(repo);
});

const deleteBtn = card.querySelector('.delete-repo-btn');
deleteBtn.addEventListener('click', async (e) => {
    e.stopPropagation();
    const name = deleteBtn.dataset.repoName;
    if (!confirm(`Delete "${name}"?\n\nThis removes the index and cloned files.`)) return;

    deleteBtn.textContent = '...';
    deleteBtn.disabled    = true;

    try {
        await apiDelete(`/api/repos/${deleteBtn.dataset.repoId}`);
        card.remove();

        if (list.querySelectorAll('.repo-card').length === 0) {
            section.classList.add('hidden');
        }
    } catch (err) {
        deleteBtn.textContent = '✕';
        deleteBtn.disabled    = false;
        alert(`Failed to delete: ${err.message}`);
    }
});
            list.appendChild(card);
        });

    } catch (e) {
    }
}



document.getElementById('analyze-btn')
        .addEventListener('click', handleAnalyzeClick);

document.getElementById('repo-url')
        .addEventListener('keydown', (e) => {
            if (e.key === 'Enter') handleAnalyzeClick();
        });

document.getElementById('repo-url')
        .addEventListener('input', hideUrlError);


loadPreviousRepos();    


function startPolling() {

    State.pollingInterval = setInterval(async () => {

        try {
            const data = await apiGet(`/api/repos/${State.repoId}/status`);
            updateIndexingUI(data);

           if (data.status === 'ready') {
    stopPolling();
    updateIndexingUI(data);

    setTimeout(() => openChatScreen(data), 700);

    const progressBar =
        document.getElementById('progress-bar-fill');

    if (progressBar) {
        progressBar.style.width = '100%';
    }

    openChatScreen(data);

} else if (data.status === 'failed') {
                stopPolling();
                showIndexingError(data.error_msg || 'Indexing failed.');
            }

        } catch (error) {

    console.error(error);

    stopPolling();

    showIndexingError(error.message);
}

    }, 2000);
}

function stopPolling() {
    if (State.pollingInterval) {
        clearInterval(State.pollingInterval);
        State.pollingInterval = null;
    }
}

    function updateIndexingUI(data) {
    const fillEl = document.getElementById('progress-bar-fill');
    if (!fillEl) return;

    const chunkCount = data.chunk_count || 0;
    const fileCount  = data.file_count  || 0;
    const status     = data.status;

    let progress = 0;
    if (status === 'ready') {
        progress = 100;
    } else if (fileCount > 0 && chunkCount > 0) {
        // Progress is estimated because the final chunk total is unknown until embedding completes.
        const estimated = fileCount * 8;
        progress = Math.min(Math.round((chunkCount / estimated) * 100), 94);
    } else if (fileCount > 0) {
        progress = 20;
    } else if (status === 'indexing') {
        progress = 5;
    }

    fillEl.style.width = `${progress}%`;

    const statsEl = document.getElementById('indexing-stats');
    if (statsEl) {
        if (status === 'ready') {
            statsEl.textContent = `${chunkCount} chunks indexed · ${fileCount} files`;
        } else if (chunkCount > 0) {
            statsEl.textContent = `${chunkCount} chunks embedded · ${fileCount} files`;
        } else if (fileCount > 0) {
            statsEl.textContent = `${fileCount} files found, embedding now...`;
        } else if (status === 'indexing') {
            statsEl.textContent = 'Cloning repository...';
        } else {
            statsEl.textContent = 'Starting...';
        }
    }

    const titleEl = document.getElementById('indexing-title');
    if (titleEl) {
        if (status === 'ready') {
            titleEl.textContent = 'Done!';
        } else if (chunkCount > 0) {
            titleEl.textContent = 'Embedding code chunks...';
        } else if (fileCount > 0) {
            titleEl.textContent = 'Starting embedding...';
        } else if (status === 'indexing') {
            titleEl.textContent = 'Cloning repository...';
        } else {
            titleEl.textContent = 'Connecting...';
        }
    }

    

    console.log('Polling update:', data);

}

function showIndexingError(message) {
   stopPolling();

    document.querySelector('.indexing-container').innerHTML = `
        <div style="text-align:center; display:flex; flex-direction:column; 
                    align-items:center; gap:16px;">
            <span style="font-size:32px;">⚠</span>
            <h2 style="color: var(--danger)">Indexing Failed</h2>
            <p style="color: var(--text-secondary)">${message}</p>
            <button class="btn btn-ghost" onclick="goBackToLanding()">
                ← Try a different repo
            </button>
        </div>
    `;
}

function goBackToLanding() {
    State.repoId   = null;
    State.repoName = null;
    showScreen('landing');
    loadPreviousRepos();
}

function openChatScreen(repoData) {


    stopPolling();

    const repoNameEl  = document.getElementById('sidebar-repo-name');
    const repoStatsEl = document.getElementById('sidebar-repo-stats');

    if (repoNameEl) {
        repoNameEl.textContent =
            State.repoName || `${repoData.owner}/${repoData.name}`;
    }

    if (repoStatsEl) {
        repoStatsEl.textContent =
            `${repoData.chunk_count || '?'} chunks · ${repoData.file_count || '?'} files`;
    }

    showScreen('chat');

    const input = document.getElementById('chat-input');

    if (input) {
        input.focus();
    }

    console.log('Chat screen opened');
}



if (!State.messages) {
    State.messages = [];
}

State.isGenerating = false;



function hideWelcomeMessage() {

    const welcome = document.getElementById('welcome-message');

    if (welcome) {
        welcome.style.display = 'none';
    }

}


function addUserMessage(text) {

    hideWelcomeMessage();

    const container = document.getElementById('chat-messages');

    if (!container) return;

    const div = document.createElement('div');

    div.className = 'message user';

    div.innerHTML = `
        <span class="message-label">You</span>

        <div class="message-bubble">
            ${escapeHtml(text)}
        </div>
    `;

    container.appendChild(div);

    State.messages.push({
        role: 'user',
        content: text,
        timestamp: Date.now()
    });

    scrollToBottom();

    return div;
}


function addThinkingMessage() {


    const container = document.getElementById('chat-messages');

    if (!container) return null;

    const div = document.createElement('div');

    const messageId = `thinking-${Date.now()}-${Math.random()
        .toString(36)
        .slice(2, 8)}`;

    div.className = 'message assistant';
    div.id        = messageId;

    div.innerHTML = `
        <span class="message-label">CodeLens</span>

        <div class="thinking">

            <div class="thinking-dots">
                <div class="thinking-dot"></div>
                <div class="thinking-dot"></div>
                <div class="thinking-dot"></div>
            </div>

            <span style="
                color: var(--text-muted);
                font-size: 12px;
            ">
                Searching codebase...
            </span>

        </div>
    `;

    container.appendChild(div);

    scrollToBottom();

    return messageId;
}



function buildSourcesHtml(sources) {

    if (!sources || sources.length === 0) {
        return '';
    }

    const tags = sources
        .filter(s => s.file)
        .slice(0, 5)
        .map((s, index) => {

            const symbol = s.symbol
                ? ` · ${escapeHtml(s.symbol)}`
                : '';

            return `
                <button 
                    class="source-tag"
                    onclick="openSourcePreview(${index})"
                >
                    ${escapeHtml(s.file)}
                    ${symbol}
                </button>
            `;
        })
        .join('');

    return `
        <div class="message-sources">
            ${tags}
        </div>
    `;
}



function attachCopyButtons(container) {

    const blocks = container.querySelectorAll('pre');

    blocks.forEach(pre => {

        const button = document.createElement('button');

        button.className = 'copy-code-btn';
        button.textContent = 'Copy';

        button.onclick = async () => {

            try {

                const code = pre.innerText;

                await navigator.clipboard.writeText(code);

                button.textContent = 'Copied';

                setTimeout(() => {
                    button.textContent = 'Copy';
                }, 1500);

            } catch {

                button.textContent = 'Failed';

            }

        };

        pre.style.position = 'relative';

        pre.appendChild(button);

    });

}



function replaceThinkingWithAnswer(
    messageId,
    answer,
    sources,
    toolCallCount
) {

    const thinkingEl = document.getElementById(messageId);

    if (!thinkingEl) {
        return;
    }

    // Sanitize generated Markdown before inserting it into the DOM.

    const renderedAnswer = DOMPurify.sanitize(
        marked.parse(answer || 'No response.')
    );

    const sourcesHtml = buildSourcesHtml(sources);

    const toolInfo = toolCallCount > 0
        ? `
            <div style="
                font-size:11px;
                color: var(--text-muted);
                padding: 6px 0 0 4px;
            ">
                ↳ ${toolCallCount} additional file${toolCallCount > 1 ? 's' : ''} read
            </div>
        `
        : '';

    thinkingEl.id = '';

    thinkingEl.innerHTML = `
        <span class="message-label">CodeLens</span>

        <div class="message-bubble assistant-bubble">
            ${renderedAnswer}
        </div>

        ${sourcesHtml}

        ${toolInfo}
    `;

    thinkingEl.querySelectorAll('pre code').forEach(block => {
        hljs.highlightElement(block);
    });

    attachCopyButtons(thinkingEl);

    State.messages.push({
        role: 'assistant',
        content: answer,
        sources,
        timestamp: Date.now()
    });

    scrollToBottom();
}



function scrollToBottom() {

    const container = document.getElementById('chat-messages');

    if (!container) return;

    container.scrollTo({
        top: container.scrollHeight,
        behavior: 'smooth'
    });

}



function escapeHtml(text) {

    const div = document.createElement('div');

    div.textContent = text;

    return div.innerHTML;

}



async function sendMessage(queryText) {


    if (State.isGenerating) {
        return;
    }

    const inputEl = document.getElementById('chat-input');

    if (!inputEl) return;

    const text = (queryText || inputEl.value).trim();

    if (!text) return;

    State.isGenerating = true;

    const sendBtn = document.getElementById('send-btn');

    inputEl.value = '';

    inputEl.style.height = 'auto';

    if (sendBtn) {
        sendBtn.disabled = true;
    }

    addUserMessage(text);

    const thinkingId = addThinkingMessage();

    try {

        const requestBody = {
            repo_id: State.repoId,
            query:   text,
            mode:    State.mode || 'chat',
        };


        if (
            State.mode === 'review' &&
            State.selectedFile
        ) {
            requestBody.target_file =
                State.selectedFile;
        }

        console.log('Sending request:', requestBody);

        const data = await apiPost(
            '/api/chat',
            requestBody
        );

        console.log('API response:', data);

        replaceThinkingWithAnswer(
            thinkingId,
            data.answer,
            data.sources || [],
            data.tool_calls_made || 0
        );

    } catch (error) {

        console.error('Chat error:', error);

        let message = 'Unknown error occurred.';

        if (
            error.message.includes('Failed to fetch')
        ) {

            message =
                'Cannot connect to backend server.';

        } else if (
            error.message.includes('timeout')
        ) {

            message =
                'Request timed out.';

        } else {

            message = error.message;

        }

        replaceThinkingWithAnswer(
            thinkingId,
            `⚠️ ${message}`,
            [],
            0
        );

    } finally {

        State.isGenerating = false;

        if (sendBtn) {
            sendBtn.disabled = false;
        }

        inputEl.focus();

    }

}



function openSourcePreview(index) {


    console.log('Open source preview:', index);

}



const sendBtn = document.getElementById('send-btn');

if (sendBtn) {

    sendBtn.addEventListener(
        'click',
        () => sendMessage()
    );

}


const chatInput = document.getElementById('chat-input');

if (chatInput) {


    chatInput.addEventListener(
        'keydown',
        (e) => {

            if (
                e.key === 'Enter' &&
                !e.shiftKey
            ) {

                e.preventDefault();

                sendMessage();

            }

        }
    );


    chatInput.addEventListener(
        'input',
        function () {

            this.style.height = 'auto';

            this.style.height =
                Math.min(this.scrollHeight, 140) + 'px';

        }
    );

}



document
    .querySelectorAll('.suggestion-chip')
    .forEach(chip => {

        chip.addEventListener('click', () => {

            if (State.isGenerating) {
                return;
            }

            const query =
                chip.dataset.query;

            sendMessage(query);

        });

    });



const newRepoBtn =
    document.getElementById('new-repo-btn');

if (newRepoBtn) {

    newRepoBtn.addEventListener(
        'click',
        goBackToLanding
    );
    document.getElementById('delete-repo-btn')
        .addEventListener('click', async () => {
            if (!State.repoId) return;
            if (!confirm(`Delete "${State.repoName}"?\n\nThis removes the index and cloned files.`)) return;

            const btn    = document.getElementById('delete-repo-btn');
            btn.textContent = '...';
            btn.disabled    = true;

            try {
                await apiDelete(`/api/repos/${State.repoId}`);
                State.repoId   = null;
                State.repoName = null;
                goBackToLanding();
            } catch (err) {
                btn.textContent = 'Delete';
                btn.disabled    = false;
                alert(`Failed to delete: ${err.message}`);
            }
        });

}


State.mode            = 'chat';
State.selectedFile    = null;
State.cachedReport    = null;
State.reportLoading   = false;



function setMode(newMode) {

    if (State.reportLoading) return;

    State.mode = newMode;

    document.querySelectorAll('.mode-tab').forEach(tab => {
        tab.classList.toggle(
            'active',
            tab.dataset.mode === newMode
        );
    });

    const filePickerSection = document.getElementById('file-picker-section');
    const suggestions       = document.getElementById('suggestions-section');
    const chatInputBar      = document.querySelector('.chat-input-bar');
    const input             = document.getElementById('chat-input');
    const reportControls =
    document.getElementById('report-controls');

    if (filePickerSection) {
    filePickerSection.classList.add('hidden');
}

if (suggestions) {
    suggestions.classList.add('hidden');
}

if (chatInputBar) {
    chatInputBar.classList.remove('hidden');
}

if (reportControls) {
    reportControls.classList.add('hidden');
}

    

    if (newMode === 'chat') {

        suggestions.classList.remove('hidden');

        input.placeholder =
            'Ask anything about this codebase...';

    }


    else if (newMode === 'review') {

       if (filePickerSection) {
    filePickerSection.classList.remove('hidden');
}

        input.placeholder =
            'Select a file above to review it...';

       

       loadFileList();

    }


    else if (newMode === 'report') {

    chatInputBar.classList.add('hidden');

    if (reportControls) {
        reportControls.classList.remove('hidden');
    }
}
}



async function loadFileList() {

    const select = document.getElementById('file-picker');

    select.innerHTML =
        '<option value="">Loading files...</option>';

    select.disabled = true;

    try {

        const data = await apiGet(
            `/api/repos/${State.repoId}/files`
        );

        const files = data.files || [];

        select.innerHTML =
            '<option value="">Select a file to review...</option>';

        files.forEach(filePath => {

            const option = document.createElement('option');

            option.value       = filePath;
            option.textContent = filePath;

            select.appendChild(option);

        });

        select.disabled = false;

    } catch (error) {

        console.error(error);

        select.innerHTML =
            '<option value="">Could not load files</option>';
    }
}



document.getElementById('file-picker')
    .addEventListener('change', function () {

        State.selectedFile = this.value;

        const input = document.getElementById('chat-input');

        if (this.value) {

            input.placeholder =
                `Ask about ${this.value} or press Enter for full review`;

        } else {

            input.placeholder =
                'Select a file above to review it...';
        }
    });



async function generateReport() {

    if (State.reportLoading) return;

    State.reportLoading = true;
   

    disableInteractiveUI();

    hideWelcomeMessage();

    const container = document.getElementById('chat-messages');
     document.querySelectorAll('.report-message')
    .forEach(el => el.remove());
    


    if (State.cachedReport) {

        renderReport(State.cachedReport);

        State.reportLoading = false;

        enableInteractiveUI();

        return;
    }


    const loadingDiv = document.createElement('div');

    loadingDiv.id = 'report-loading';

    loadingDiv.className = 'message assistant';

    loadingDiv.innerHTML = `
        <div class="message-bubble">
            <div style="
                display:flex;
                flex-direction:column;
                align-items:center;
                gap:14px;
                padding:20px;
            ">
                <span class="spinner"></span>

                <div style="font-weight:600;">
                    Generating Repository Intelligence Report
                </div>

                <div style="
                    color: var(--text-secondary);
                    font-size: 14px;
                    text-align:center;
                    line-height:1.6;
                ">
                    Reading architecture, components,
                    patterns, and dependencies...
                </div>
            </div>
        </div>
    `;

    container.appendChild(loadingDiv);

    scrollToBottom();


    try {

      const data = await apiPost('/api/report', {
    repo_id: State.repoId
});

        document.getElementById('report-loading')?.remove();

        State.cachedReport = data.report;

        renderReport(data.report);

    } catch (error) {

        console.error(error);

        document.getElementById('report-loading')?.remove();

        const errDiv = document.createElement('div');

        errDiv.className = 'message assistant';

        errDiv.innerHTML = `
            <div class="message-bubble" style="
                color: var(--danger);
            ">
                Failed to generate report.<br><br>
                ${error.message}
            </div>
        `;

        container.appendChild(errDiv);

        scrollToBottom();

    } finally {

        State.reportLoading = false;

        enableInteractiveUI();
    }
}



function renderReport(reportMarkdown) {

    const container = document.getElementById('chat-messages');

    const reportDiv = document.createElement('div');

    reportDiv.className =
    'message assistant report-message';

    reportDiv.innerHTML = `
        <span class="message-label">
            Repository Intelligence Report
        </span>

        <div class="message-bubble report-bubble">
            ${marked.parse(reportMarkdown)}
        </div>
    `;

    container.appendChild(reportDiv);

    reportDiv.querySelectorAll('pre code')
        .forEach(block => {
            hljs.highlightElement(block);
        });

    scrollToBottom();
}



function disableInteractiveUI() {

    document.querySelectorAll('.mode-tab')
        .forEach(btn => btn.disabled = true);

    const sendBtn = document.getElementById('send-btn');

    if (sendBtn) {
        sendBtn.disabled = true;
    }
}

function enableInteractiveUI() {

    document.querySelectorAll('.mode-tab')
        .forEach(btn => btn.disabled = false);

    const sendBtn = document.getElementById('send-btn');

    if (sendBtn) {
        sendBtn.disabled = false;
    }
}



document.getElementById('generate-report-btn')
    ?.addEventListener('click', generateReport);



document.querySelectorAll('.mode-tab')
    .forEach(tab => {

        tab.addEventListener('click', () => {

            setMode(tab.dataset.mode);

        });

    });
