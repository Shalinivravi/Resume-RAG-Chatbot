document.addEventListener('DOMContentLoaded', () => {
    // Game State
    let grid = Array(9).fill().map(() => Array(9).fill(0));
    let solution = Array(9).fill().map(() => Array(9).fill(0));
    let initialMask = Array(9).fill().map(() => Array(9).fill(false)); // true if initial
    let selectedCell = null; // {r, c}
    let difficulty = 'medium';
    let mistakes = 0;
    let timer = 0;
    let timerInterval = null;
    let history = []; // Stack for undo
    let gameActive = false;

    // Elements
    const boardElement = document.getElementById('game-board');
    const diffDisplay = document.getElementById('diff-display');
    const mistakesDisplay = document.getElementById('mistakes-count');
    const timerDisplay = document.getElementById('timer');
    const diffBtns = document.querySelectorAll('.diff-btn');
    const btnNewGame = document.getElementById('btn-new-game');
    const btnUndo = document.getElementById('btn-undo');
    const btnErase = document.getElementById('btn-erase');
    const btnHint = document.getElementById('btn-hint');
    const numpads = document.querySelectorAll('.number-btn');
    const modal = document.getElementById('win-modal');
    const finalTimeDisplay = document.getElementById('final-time');
    const btnPlayAgain = document.getElementById('btn-play-again');

    // Init
    initGame();

    // Event Listeners
    btnNewGame.addEventListener('click', () => startNewGame(difficulty));
    btnPlayAgain.addEventListener('click', () => {
        closeModal();
        startNewGame(difficulty);
    });

    diffBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            diffBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            difficulty = btn.dataset.level;
            startNewGame(difficulty);
        });
    });

    numpads.forEach(btn => {
        btn.addEventListener('click', () => {
            const num = parseInt(btn.dataset.number);
            handleInput(num);
        });
    });

    // Keyboard support
    document.addEventListener('keydown', (e) => {
        if (!gameActive) return;

        if (e.key >= '1' && e.key <= '9') {
            handleInput(parseInt(e.key));
        } else if (e.key === 'Backspace' || e.key === 'Delete') {
            handleErase();
        } else if (e.key.startsWith('Arrow')) {
            moveSelection(e.key);
        }
    });

    btnErase.addEventListener('click', handleErase);
    btnUndo.addEventListener('click', handleUndo);

    // Hint: Find a random empty cell and fill it
    btnHint.addEventListener('click', () => {
        if (!gameActive) return;

        // Find all empty cells
        const emptyCells = [];
        for (let r = 0; r < 9; r++) {
            for (let c = 0; c < 9; c++) {
                if (grid[r][c] === 0) {
                    emptyCells.push({ r, c });
                }
            }
        }

        if (emptyCells.length > 0) {
            const randomCell = emptyCells[Math.floor(Math.random() * emptyCells.length)];
            // Fill it with correct value from solution
            const val = solution[randomCell.r][randomCell.c];
            fillCell(randomCell.r, randomCell.c, val);
            highlightRelated(randomCell.r, randomCell.c);
        }
    });


    // Functions
    function initGame() {
        startNewGame(difficulty);
    }

    function startNewGame(diff) {
        clearInterval(timerInterval);
        timer = 0;
        mistakes = 0;
        history = [];
        updateTimerDisplay();
        updateMistakesDisplay();

        diffDisplay.textContent = diff.charAt(0).toUpperCase() + diff.slice(1);

        generateBoard();
        createBoardUI();

        gameActive = true;
        startTimer();
    }

    function generateBoard() {
        // Simple generation: Fill diagonal boxes -> Solve -> Remove elements
        // 1. Clear array
        grid = Array(9).fill().map(() => Array(9).fill(0));

        // 2. Fill diagonal 3x3 matrices
        fillDiagonal();

        // 3. Solve completely to get solution
        solveSudoku(grid);
        solution = grid.map(row => [...row]); // Copy solution

        // 4. Remove Digits based on difficulty
        removeDigits(difficulty);

        // Mark initial filled cells
        initialMask = grid.map(row => row.map(val => val !== 0));
    }

    function fillDiagonal() {
        for (let i = 0; i < 9; i = i + 3) {
            fillBox(i, i);
        }
    }

    function fillBox(row, col) {
        let num;
        for (let i = 0; i < 3; i++) {
            for (let j = 0; j < 3; j++) {
                do {
                    num = Math.floor(Math.random() * 9) + 1;
                } while (!isSafeInBox(row, col, num));
                grid[row + i][col + j] = num;
            }
        }
    }

    function isSafeInBox(rowStart, colStart, num) {
        for (let i = 0; i < 3; i++) {
            for (let j = 0; j < 3; j++) {
                if (grid[rowStart + i][colStart + j] === num) {
                    return false;
                }
            }
        }
        return true;
    }

    function solveSudoku(board) {
        let row = -1;
        let col = -1;
        let isEmpty = false;
        for (let i = 0; i < 9; i++) {
            for (let j = 0; j < 9; j++) {
                if (board[i][j] === 0) {
                    row = i;
                    col = j;
                    isEmpty = true;
                    break;
                }
            }
            if (isEmpty) {
                break;
            }
        }
        if (!isEmpty) {
            return true; // Solved
        }

        for (let num = 1; num <= 9; num++) {
            if (isSafe(board, row, col, num)) {
                board[row][col] = num;
                if (solveSudoku(board)) {
                    return true;
                }
                board[row][col] = 0;
            }
        }
        return false;
    }

    function isSafe(board, row, col, num) {
        // Row check
        for (let x = 0; x < 9; x++) {
            if (board[row][x] === num) return false;
        }
        // Col check
        for (let x = 0; x < 9; x++) {
            if (board[x][col] === num) return false;
        }
        // Box check
        let startRow = row - row % 3;
        let startCol = col - col % 3;
        for (let i = 0; i < 3; i++) {
            for (let j = 0; j < 3; j++) {
                if (board[i + startRow][j + startCol] === num) return false;
            }
        }
        return true;
    }

    function removeDigits(diff) {
        let attempts = 5;
        let count = 30; // Amount to remove
        if (diff === 'easy') count = 30;
        if (diff === 'medium') count = 45;
        if (diff === 'hard') count = 55;

        while (count > 0) {
            let row = Math.floor(Math.random() * 9);
            let col = Math.floor(Math.random() * 9);
            while (grid[row][col] === 0) {
                row = Math.floor(Math.random() * 9);
                col = Math.floor(Math.random() * 9);
            }
            grid[row][col] = 0;
            count--;
        }
    }

    function createBoardUI() {
        boardElement.innerHTML = '';
        for (let i = 0; i < 9; i++) {
            for (let j = 0; j < 9; j++) {
                const cell = document.createElement('div');
                cell.classList.add('cell');
                cell.dataset.row = i;
                cell.dataset.col = j;

                if (grid[i][j] !== 0) {
                    cell.textContent = grid[i][j];
                    cell.classList.add('initial');
                }

                // Borders for 3x3 (handled in CSS mostly)

                cell.addEventListener('click', () => selectCell(i, j));
                boardElement.appendChild(cell);
            }
        }
    }

    function selectCell(row, col) {
        if (!gameActive) return;

        selectedCell = { r: row, c: col };

        // Visual Logic
        document.querySelectorAll('.cell').forEach(c => {
            c.classList.remove('selected', 'related', 'highlight-number');
        });

        const cells = document.querySelectorAll('.cell');
        const index = row * 9 + col;
        cells[index].classList.add('selected');

        // Highlight row, col, box
        const startRow = row - row % 3;
        const startCol = col - col % 3;

        for (let i = 0; i < 9; i++) {
            // Row
            cells[row * 9 + i].classList.add('related');
            // Col
            cells[i * 9 + col].classList.add('related');
        }
        // Box
        for (let i = 0; i < 3; i++) {
            for (let j = 0; j < 3; j++) {
                cells[(startRow + i) * 9 + (startCol + j)].classList.add('related');
            }
        }

        // Highlight same numbers
        const currentVal = grid[row][col];
        if (currentVal !== 0) {
            for (let r = 0; r < 9; r++) {
                for (let c = 0; c < 9; c++) {
                    if (grid[r][c] === currentVal) {
                        cells[r * 9 + c].classList.add('highlight-number');
                    }
                }
            }
        }
    }

    function moveSelection(key) {
        if (!selectedCell) return;
        let { r, c } = selectedCell;

        if (key === 'ArrowUp') r = Math.max(0, r - 1);
        if (key === 'ArrowDown') r = Math.min(8, r + 1);
        if (key === 'ArrowLeft') c = Math.max(0, c - 1);
        if (key === 'ArrowRight') c = Math.min(8, c + 1);

        selectCell(r, c);
    }

    function handleInput(num) {
        if (!selectedCell || !gameActive) return;
        const { r, c } = selectedCell;

        if (initialMask[r][c]) return; // Can't change initial cells

        // Undo support
        history.push({
            r, c,
            prevVal: grid[r][c],
            newVal: num,
            type: 'input'
        });

        fillCell(r, c, num);
    }

    function fillCell(r, c, num) {
        grid[r][c] = num;

        const cell = document.querySelector(`.cell[data-row="${r}"][data-col="${c}"]`);
        cell.textContent = num;
        cell.classList.add('filled');
        cell.classList.remove('error');

        // Validation check for this move
        if (num !== solution[r][c]) {
            cell.classList.add('error');
            mistakes++;
            updateMistakesDisplay();

            if (mistakes >= 3) {
                // Optional: Game Over Logic, or just let them play
                // alert('3 Mistakes! Be careful.');
            }
        }

        // Re-highlight usually to show new number highlighting
        selectCell(r, c);

        checkWin();
    }

    function handleErase() {
        if (!selectedCell || !gameActive) return;
        const { r, c } = selectedCell;
        if (initialMask[r][c]) return;

        history.push({
            r, c,
            prevVal: grid[r][c],
            newVal: 0,
            type: 'erase'
        });

        grid[r][c] = 0;
        const cell = document.querySelector(`.cell[data-row="${r}"][data-col="${c}"]`);
        cell.textContent = '';
        cell.classList.remove('filled', 'error');

        selectCell(r, c);
    }

    function handleUndo() {
        if (history.length === 0 || !gameActive) return;
        const lastAction = history.pop();
        const { r, c, prevVal } = lastAction;

        grid[r][c] = prevVal;
        const cell = document.querySelector(`.cell[data-row="${r}"][data-col="${c}"]`);

        if (prevVal === 0) {
            cell.textContent = '';
            cell.classList.remove('filled', 'error');
        } else {
            cell.textContent = prevVal;
            cell.classList.add('filled');
            if (prevVal !== solution[r][c]) {
                cell.classList.add('error');
            } else {
                cell.classList.remove('error');
            }
        }
        selectCell(r, c);
    }

    function startTimer() {
        timerInterval = setInterval(() => {
            timer++;
            updateTimerDisplay();
        }, 1000);
    }

    function updateTimerDisplay() {
        const min = Math.floor(timer / 60).toString().padStart(2, '0');
        const sec = (timer % 60).toString().padStart(2, '0');
        timerDisplay.textContent = `${min}:${sec}`;
    }

    function updateMistakesDisplay() {
        mistakesDisplay.textContent = `${mistakes}/3`;
        if (mistakes >= 3) {
            mistakesDisplay.style.color = 'var(--error-color)';
        } else {
            mistakesDisplay.style.color = 'var(--text-color)';
        }
    }

    function checkWin() {
        // Simple check: is grid full and no errors?
        for (let i = 0; i < 9; i++) {
            for (let j = 0; j < 9; j++) {
                if (grid[i][j] === 0) return; // Not full
                if (grid[i][j] !== solution[i][j]) return; // Error exists
            }
        }

        gameWin();
    }

    function gameWin() {
        gameActive = false;
        clearInterval(timerInterval);
        finalTimeDisplay.textContent = timerDisplay.textContent;
        // Confetti effect or something could go here

        modal.classList.remove('hidden');
        setTimeout(() => modal.classList.add('visible'), 100);
    }

    function closeModal() {
        modal.classList.remove('visible');
        setTimeout(() => modal.classList.add('hidden'), 300);
    }
});
