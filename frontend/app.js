const API_BASE_URL = window.location.hostname === '127.0.0.1' || window.location.hostname === 'localhost' 
    ? '' 
    : 'https://fusion-mail-app.onrender.com';

const state = {
    emails: [
        {
            id: 1,
            subject: "URGENT: Server Down in Production",
            body: "The main production server has crashed! We are losing customers every second. Someone needs to look at this IMMEDIATELY!!! This is a disaster.",
            timestamp: "2023-10-27T08:15:00",
            sender: "DevOps Team"
        },
        {
            id: 2,
            subject: "Weekly Newsletter",
            body: "Here is the weekly update on company activities. We had a great picnic last Friday. Hope you all enjoyed it.",
            timestamp: "2023-10-27T09:00:00",
            sender: "HR Department"
        },
        {
            id: 3,
            subject: "Client Meeting Reschedule?",
            body: "I am really worried about the client meeting tomorrow. I don't feel prepared at all. Can we please push it back? I'm stressing out.",
            timestamp: "2023-10-27T18:30:00", // Off hours
            sender: "Junior Dev"
        },
        {
            id: 4,
            subject: "Great work on the Q3 Report",
            body: "Just wanted to say fantastic job on the Q3 report. The board is very happy with the results! Keep it up!",
            timestamp: "2023-10-26T14:20:00",
            sender: "CEO"
        },
        {
            id: 5,
            subject: "Invoice #12345",
            body: "Please find attached the invoice for the recent services. Payment is due in 30 days.",
            timestamp: "2023-10-27T10:00:00",
            sender: "Vendor Inc"
        },
        {
            id: 6,
            subject: "Where is the report???",
            body: "I have been waiting for the report for 2 hours. WHY IS IT NOT HERE YET?! I AM VERY ANGRY.",
            timestamp: "2023-10-27T11:00:00",
            sender: "Manager"
        }
    ],
    processedEmails: [],
    sortByUrgency: false,
    searchTerm: "",
    activeFilter: "all"
};

const dom = {
    emailList: document.getElementById('email-list'),
    urgencyToggle: document.getElementById('urgency-toggle'),
    urgencyState: document.getElementById('urgency-state'),
    inboxCount: document.getElementById('inbox-count'),
    gmailSync: document.getElementById('gmail-sync'),
    searchInput: document.querySelector('.search-bar input'),
    filterTabs: document.querySelectorAll('.filters .filter'),
    sidebarItems: document.querySelectorAll('.sidebar li')
};

async function init() {
    renderLoading();
    
    // Add Event Listeners
    setupEventListeners();

    try {
        // Send emails to backend for AI analysis
        const response = await fetch(`${API_BASE_URL}/api/analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ emails: state.emails })
        });
        
        const data = await response.json();
        
        // Merge results with original emails
        state.processedEmails = state.emails.map(email => {
            const analysis = data.results.find(r => r.id === email.id);
            // Use backend provided unread status if available, else default false
            const isUnread = email.isUnread !== undefined ? email.isUnread : (analysis?.isUnread || false);
            return { ...email, ...analysis, isUnread };
        });
        
        // Initial Sort (by timestamp default, or just order)
        state.processedEmails.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
        
        renderEmails();
        updateStats();
        
    } catch (error) {
        console.error("Error analyzing emails:", error);
        dom.emailList.innerHTML = `<div class="loading" style="color:red">Error connecting to AI Backend. Ensure Flask is running.</div>`;
    }
}

function renderLoading() {
    dom.emailList.innerHTML = `<div class="loading"><i class="fa-solid fa-circle-notch fa-spin"></i> Analyzing Emotions & Urgency...</div>`;
}

function setupEventListeners() {
    // Search
    dom.searchInput.addEventListener('input', (e) => {
        state.searchTerm = e.target.value.toLowerCase();
        renderEmails();
    });

    // Filter Tabs
    dom.filterTabs.forEach(tab => {
        tab.addEventListener('click', (e) => {
            // Remove active from all
            dom.filterTabs.forEach(t => t.classList.remove('active'));
            // Add to clicked
            e.target.classList.add('active');
            
            // Set state
            const text = e.target.innerText.toLowerCase();
            if (text === 'all') state.activeFilter = 'all';
            else if (text === 'unread') state.activeFilter = 'unread';
            else if (text === 'urgent') state.activeFilter = 'urgent';
            
            renderEmails();
        });
    });

    // Sidebar Items
    dom.sidebarItems.forEach(item => {
        item.addEventListener('click', () => {
             dom.sidebarItems.forEach(i => i.classList.remove('active'));
             item.classList.add('active');
             
             if (item.innerText.includes('Inbox')) {
                 renderEmails();
             } else {
                 dom.emailList.innerHTML = `<div class="loading"><i class="fa-solid fa-hammer"></i> ${item.innerText.trim()} Feature Under Construction</div>`;
             }
        });
    });

    // Urgency Toggle
    dom.urgencyToggle.addEventListener('click', () => {
        state.sortByUrgency = !state.sortByUrgency;
        if (state.sortByUrgency) {
            dom.urgencyToggle.classList.remove('off');
            dom.urgencyState.innerText = "ON";
        } else {
            dom.urgencyToggle.classList.add('off');
            dom.urgencyState.innerText = "OFF";
        }
        renderEmails();
    });

    // Gmail Sync
    dom.gmailSync.addEventListener('click', async () => {
        dom.emailList.innerHTML = `<div class="loading"><i class="fa-brands fa-google fa-bounce"></i> Connecting to Gmail...</div>`;
        
        try {
            const response = await fetch(`${API_BASE_URL}/api/sync`, { method: 'POST' });
            const data = await response.json();
            
            if (data.status === 'no_emails') {
                 alert(data.message);
                 renderEmails(); // Restore old list
                 return;
            }
            
            if (data.error) {
                alert("Error: " + data.error);
                renderEmails();
                return;
            }
            
            // Add new emails to top
            // Backend now provides isUnread status from Gmail
            const newEmails = data.results; 
            state.processedEmails = [...newEmails, ...state.processedEmails];
            state.emails = [...newEmails, ...state.emails];
            
             // Re-sort
            state.processedEmails.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
            
            renderEmails();
            updateStats();
            
        } catch (error) {
            console.error("Sync failed:", error);
            alert("Failed to sync. Check console.");
            renderEmails();
        }
    });

}

function renderEmails() {
    dom.emailList.innerHTML = '';
    
    // Filter
    let displayList = state.processedEmails.filter(email => {
        // Search Filter
        const matchesSearch = 
            email.subject.toLowerCase().includes(state.searchTerm) || 
            email.sender.toLowerCase().includes(state.searchTerm) ||
            email.body.toLowerCase().includes(state.searchTerm);
        
        if (!matchesSearch) return false;

        // Tab Filter
        if (state.activeFilter === 'unread' && !email.isUnread) return false;
        if (state.activeFilter === 'urgent' && email.urgency_score < 0.7) return false;

        return true;
    });

    // Sort
    if (state.sortByUrgency) {
        displayList.sort((a, b) => b.urgency_score - a.urgency_score);
    } // else default sort is by time (preserved from array order usually)
    
    if (displayList.length === 0) {
        dom.emailList.innerHTML = `<div class="loading">No emails found.</div>`;
        return;
    }

    displayList.forEach(email => {
        const el = document.createElement('div');
        el.className = 'email-item';
        // Bold if unread
        if (email.isUnread) el.style.fontWeight = "600";
        
        // Color mapping
        let dotColor = 'var(--emotion-neutral)';
        let emotionClass = 'neutral';
        if (email.emotion === 'Angry') { dotColor = 'var(--emotion-angry)'; emotionClass = 'angry'; }
        if (email.emotion === 'Anxious') { dotColor = 'var(--emotion-anxious)'; emotionClass = 'anxious'; }
        if (email.emotion === 'Happy') { dotColor = 'var(--emotion-happy)'; emotionClass = 'happy'; }
        
        // Urgency Bar Height %
        const urgencyHeight = Math.max(10, email.urgency_score * 100);
        
        el.innerHTML = `
            <div class="status-indicator">
                <div class="emotion-dot" style="background-color: ${dotColor}"></div>
                <div class="urgency-bar" title="Urgency: ${(email.urgency_score * 10).toFixed(1)}/10">
                    <div class="urgency-fill" style="height: ${urgencyHeight}%"></div>
                </div>
            </div>
            
            <div class="email-content">
                <div class="sender">${email.sender}</div>
                <div class="subject">${email.subject}</div>
                <div class="preview">${email.body}</div>
            </div>
            
            <div class="email-meta">
                <span>${new Date(email.timestamp).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}</span>
                <span class="emotion-tag ${emotionClass}">${email.emotion}</span>
            </div>

            <!-- AI Summary Overlay -->
            <div class="ai-summary">
                <div class="summary-title"><i class="fa-solid fa-robot"></i> AI Summary</div>
                <div class="summary-text">"${email.summary}"</div>
                <div style="margin-top:0.5rem; font-size:0.8rem; color:${dotColor}">
                    Detected: ${email.emotion} ${(email.urgency_score*100).toFixed(0)}% Urgency
                </div>
            </div>
        `;
        
        dom.emailList.appendChild(el);
    });
}

function updateStats() {
    dom.inboxCount.innerText = state.emails.length;
}

// Start
init();
