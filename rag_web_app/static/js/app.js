// 通用工具函数
class RagChatApp {
    constructor() {
        this.sessionId = null;
        this.isStreaming = false;
        this.initEventListeners();
        this.loadHistory();
    }

    initEventListeners() {
        // 登录表单
        const loginForm = document.getElementById('loginForm');
        if (loginForm) {
            loginForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.handleLogin();
            });
        }

        // 聊天表单
        const chatForm = document.getElementById('chatForm');
        if (chatForm) {
            chatForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.handleMessageSubmit();
            });
        }

        // 清空历史按钮
        const clearBtn = document.getElementById('clearBtn');
        if (clearBtn) {
            clearBtn.addEventListener('click', () => this.clearHistory());
        }

        // 新对话按钮
        const newChatBtn = document.getElementById('newChatBtn');
        if (newChatBtn) {
            newChatBtn.addEventListener('click', () => this.startNewChat());
        }

        // 退出登录按钮
        const logoutBtn = document.getElementById('logoutBtn');
        if (logoutBtn) {
            logoutBtn.addEventListener('click', () => this.logout());
        }

        // 输入框回车发送
        const userInput = document.getElementById('userInput');
        if (userInput) {
            userInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    this.handleMessageSubmit();
                }
            });
        }
    }

    async handleLogin() {
        const usernameInput = document.getElementById('username');
        const username = usernameInput.value.trim();

        if (!username) {
            alert('请输入用户名');
            return;
        }

        try {
            const formData = new FormData();
            formData.append('username', username);

            const response = await fetch('/login', {
                method: 'POST',
                body: formData
            });

            if (response.ok) {
                const data = await response.json();
                window.location.href = data.redirect;
            } else {
                const error = await response.json();
                alert(error.detail || '登录失败，请重试');
            }
        } catch (error) {
            console.error('登录错误:', error);
            alert('网络错误，请检查连接');
        }
    }

    async handleMessageSubmit() {
        const userInput = document.getElementById('userInput');
        const message = userInput.value.trim();

        if (!message) return;
        if (this.isStreaming) return;

        // 获取 session_id
        let sessionId = '';
        if (typeof SESSION_ID !== 'undefined') {
            sessionId = SESSION_ID;
        } else {
            const urlParams = new URLSearchParams(window.location.search);
            sessionId = urlParams.get('session_id') || '';
        }

        if (!sessionId) {
            alert('会话无效，请重新登录');
            window.location.href = '/';
            return;
        }

        this.sessionId = sessionId;

        // 添加用户消息到界面
        this.addMessage(message, 'user');
        userInput.value = '';

        // 显示正在输入指示器
        this.showTypingIndicator(true);
        this.isStreaming = true;

        try {
            // 创建 AI 消息占位符
            const aiMessageId = this.addMessage('', 'assistant');

            // 发送请求并处理流式响应
            await this.streamResponse(message, sessionId, aiMessageId);

        } catch (error) {
            console.error('发送消息错误:', error);
            this.addMessage('抱歉，出现了错误，请稍后重试。', 'assistant');
        } finally {
            this.showTypingIndicator(false);
            this.isStreaming = false;
            userInput.focus();
        }
    }

    async streamResponse(message, sessionId, messageElement) {
        const formData = new FormData();
        formData.append('session_id', sessionId);
        formData.append('query', message);

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`HTTP error! status: ${response.status}, body: ${errorText}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            let aiResponse = '';

            while (true) {
                const { done, value } = await reader.read();

                if (done) {
                    break;
                }

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n\n');
                buffer = lines.pop(); // 剩余的不完整数据

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const data = line.slice(6);
                        if (data) {
                            try {
                                const parsed = JSON.parse(data);

                                if (parsed.chunk) {
                                    aiResponse += parsed.chunk;
                                    this.updateMessage(messageElement, aiResponse);
                                    // 自动滚动到底部
                                    this.scrollToBottom();
                                }

                                if (parsed.error) {
                                    this.updateMessage(messageElement, parsed.error);
                                }

                                if (parsed.done) {
                                    break;
                                }
                            } catch (e) {
                                console.error('解析 SSE 数据错误:', e);
                            }
                        }
                    }
                }
            }

        } catch (error) {
            console.error('流式响应错误:', error);
            this.updateMessage(messageElement, '抱歉，出现了网络错误，请稍后重试。');
        }
    }

    addMessage(content, role) {
        const messagesContainer = document.getElementById('messagesContainer');

        // 移除欢迎消息（如果是第一条消息）
        const welcomeMessage = messagesContainer.querySelector('.welcome-message');
        if (welcomeMessage && role === 'user') {
            welcomeMessage.remove();
        }

        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;
        messageDiv.id = `msg-${Date.now()}`;

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.textContent = content;

        const timeDiv = document.createElement('div');
        timeDiv.className = 'message-timestamp';
        timeDiv.textContent = this.formatTime(new Date());

        messageDiv.appendChild(contentDiv);
        messageDiv.appendChild(timeDiv);
        messagesContainer.appendChild(messageDiv);

        this.scrollToBottom();
        return messageDiv;
    }

    updateMessage(messageElement, content) {
        const contentDiv = messageElement.querySelector('.message-content');
        if (contentDiv) {
            contentDiv.textContent = content;
        }
    }

    showTypingIndicator(show) {
        const indicator = document.getElementById('typingIndicator');
        if (indicator) {
            indicator.style.display = show ? 'flex' : 'none';
        }
    }

    async loadHistory() {
        let sessionId = '';
        if (typeof SESSION_ID !== 'undefined') {
            sessionId = SESSION_ID;
        } else {
            const urlParams = new URLSearchParams(window.location.search);
            sessionId = urlParams.get('session_id') || '';
        }

        if (!sessionId) return;

        try {
            const response = await fetch(`/api/history?session_id=${encodeURIComponent(sessionId)}`);
            if (response.ok) {
                const data = await response.json();
                if (data.messages && data.messages.length > 0) {
                    // 清空欢迎消息
                    const welcomeMessage = document.querySelector('.welcome-message');
                    if (welcomeMessage) {
                        welcomeMessage.remove();
                    }

                    // 添加历史消息
                    data.messages.forEach(msg => {
                        this.addMessage(msg.content, msg.role);
                    });

                    this.scrollToBottom();
                }
            }
        } catch (error) {
            console.error('加载历史记录错误:', error);
        }
    }

    async clearHistory() {
        if (!confirm('确定要清空当前对话历史吗？')) {
            return;
        }

        if (!this.sessionId) {
            if (typeof SESSION_ID !== 'undefined') {
                this.sessionId = SESSION_ID;
            } else {
                const urlParams = new URLSearchParams(window.location.search);
                this.sessionId = urlParams.get('session_id') || '';
            }
        }

        if (!this.sessionId) return;

        try {
            const formData = new FormData();
            formData.append('session_id', this.sessionId);

            const response = await fetch('/api/clear', {
                method: 'POST',
                body: formData
            });

            if (response.ok) {
                // 清空消息界面
                const messagesContainer = document.getElementById('messagesContainer');
                messagesContainer.innerHTML = `
                    <div class="welcome-message">
                        <h3>👋 欢迎使用 RAG 智能助手！</h3>
                        <p>我是基于检索增强生成技术的智能助手，可以回答各种专业问题。</p>
                        <p>请在下方的输入框中提问，我会尽力为您解答。</p>
                    </div>
                `;

                alert('历史记录已清空');
            } else {
                const error = await response.json();
                alert(error.detail || '清空失败，请重试');
            }
        } catch (error) {
            console.error('清空历史记录错误:', error);
            alert('清空失败，请重试');
        }
    }

    startNewChat() {
        if (confirm('开始新对话会清空当前历史，确定吗？')) {
            this.clearHistory();
        }
    }

    async logout() {
        if (!confirm('确定要退出登录吗？')) {
            return;
        }

        try {
            const response = await fetch('/api/logout', {
                method: 'POST'
            });

            if (response.ok) {
                window.location.href = '/';
            }
        } catch (error) {
            console.error('退出登录错误:', error);
            window.location.href = '/';
        }
    }

    scrollToBottom() {
        const messagesContainer = document.getElementById('messagesContainer');
        if (messagesContainer) {
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }
    }

    formatTime(date) {
        return date.toLocaleTimeString('zh-CN', {
            hour: '2-digit',
            minute: '2-digit'
        });
    }
}

// 页面加载完成后初始化应用
document.addEventListener('DOMContentLoaded', () => {
    window.ragChatApp = new RagChatApp();
});