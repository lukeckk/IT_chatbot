class Chatbox {
    constructor() {
        this.args = {
            chatBox: document.querySelector('.chatbox__support'),
            sendButton: document.querySelector('.send__button')
        }

        this.state = false;
        this.messages = [];
    }

    display() {
        const { chatBox, sendButton } = this.args;

        sendButton.addEventListener('click', () => this.onSendButton(chatBox));

        const node = chatBox.querySelector('input');
        node.addEventListener("keyup", ({ key }) => {
            if (key === "Enter") {
                this.onSendButton(chatBox);
            }
        });
    }

    onSendButton(chatbox) {
        var textField = chatbox.querySelector('input');
        let text1 = textField.value;
        if (text1 === "") {
            return;
        }

        let msg1 = { name: "User", message: text1 };
        this.messages.push(msg1);
        textField.value = ''; // Clear the input field immediately
        this.updateChatText(chatbox);

        this.showTypingIndicator(chatbox);

        setTimeout(() => {
            fetch('http://127.0.0.1:5000/predict', {
                method: 'POST',
                body: JSON.stringify({ message: text1 }),
                mode: 'cors',
                headers: {
                    'Content-Type': 'application/json'
                },
            })
            .then(r => r.json())
            .then(r => {
                let msg2 = { name: "IT Helper", message: r.answer };
                this.messages.push(msg2);
                this.updateChatText(chatbox);
            }).catch((error) => {
                console.error('Error:', error);
                this.updateChatText(chatbox);
            }).finally(() => {
                this.hideTypingIndicator(chatbox);
            });
        }, 500); // Show typing GIF for 2 seconds
    }

    showTypingIndicator(chatbox) {
        const chatmessage = chatbox.querySelector('.chatbox__messages');
        const typingIndicator = document.createElement('div');
        typingIndicator.className = 'typing-indicator';
        typingIndicator.innerHTML = '<img src="https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExbGljeTB5NzAxdjduN3Ruc25rbDdmNDF1anRsNHl4OWQ5ZTl3YXF0ZyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9cw/Sv8gUHtkqgylj9sEvS/giphy.webp" alt="Typing..." class="typing-image">';
        chatmessage.appendChild(typingIndicator); // Ensure typing indicator is at the bottom
    }

    hideTypingIndicator(chatbox) {
        const chatmessage = chatbox.querySelector('.chatbox__messages');
        const typingIndicator = chatmessage.querySelector('.typing-indicator');
        if (typingIndicator) {
            chatmessage.removeChild(typingIndicator);
        }
    }

    updateChatText(chatbox) {
        var html = '';
        this.messages.forEach(function(item, index) {
            if (item.name === "IT Helper") {
                html += '<div class="messages__item messages__item--operator">' + item.message + '</div>';
            } else {
                html += '<div class="messages__item messages__item--visitor">' + item.message + '</div>';
            }
        });

        const chatmessage = chatbox.querySelector('.chatbox__messages');
        chatmessage.innerHTML = html;

        // Ensure typing indicator remains at the bottom
        if (chatbox.querySelector('.typing-indicator')) {
            this.showTypingIndicator(chatbox);
        }
    }
}

const chatbox = new Chatbox();
chatbox.display();
