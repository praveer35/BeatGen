let melody = [
    [0, 0.0, 1.0],
    [2, 1.0, 3.0],
    [3, 3.0, 6.0]
];

const dawContainer = document.getElementById('daw-container');
let draggedNote = null;
let resizeHandle = null;
let startX, startY;
let initialLeft, initialTop, initialWidth;
let isDragging = false;
let isResizing = false;

dawContainer.addEventListener('mousedown', (e) => {
    if (e.target.classList.contains('resize-handle')) {
        isResizing = true;
        resizeHandle = e.target;
        draggedNote = resizeHandle.parentElement;
        startX = e.clientX;
        initialWidth = draggedNote.offsetWidth;
        e.preventDefault();
    } else if (e.target.classList.contains('note')) {
        isDragging = true;
        draggedNote = e.target;
        startX = e.clientX;
        startY = e.clientY;
        initialLeft = parseFloat(draggedNote.style.left) || 0;
        initialTop = parseFloat(draggedNote.style.top) || 0;
        e.preventDefault();
    }
});

document.addEventListener('mousemove', (e) => {
    if (isResizing && draggedNote) {
        const deltaX = e.clientX - startX;

        // Prevent the width from going below a certain value (e.g., 1 beat)
        const minWidth = 25;
        let newWidth = Math.max(initialWidth + deltaX, minWidth) - 25;

        // Calculate new end time based on resized width
        const startTime = parseFloat(draggedNote.dataset.start);
        const endTime = startTime + newWidth / 25; // Assuming 25px = 1.0 beat

        // Snap to grid (round to nearest 0.25 beats)
        const snappedEndTime = Math.round(endTime);

        const maxWidth = dawContainer.clientWidth - parseFloat(draggedNote.style.left) - 22;
        newWidth = (snappedEndTime - startTime) * 25 + 4;
        newWidth = Math.min(newWidth, maxWidth);
        const newSnappedEndTime = Math.round(startTime + (newWidth - 4)/25);

        // Update note width 
        draggedNote.style.width = `${newWidth}px`;
        draggedNote.dataset.end = parseInt(newSnappedEndTime);
    } else if (isDragging && draggedNote) {
        const deltaX = e.clientX - startX;
        const deltaY = e.clientY - startY;

        // Calculate new positions
        let newLeft = initialLeft + deltaX;
        let newTop = initialTop + deltaY;

        // Snap to grid
        const snappedLeft = Math.round(newLeft / 25) * 25;
        const snappedTop = Math.round(newTop / 25) * 25;

        // Calculate new start time based on drag position
        const startTime = snappedLeft / 25; // Assuming 25px = 1.0 beat
        const endTime = startTime + (parseFloat(draggedNote.dataset.end) - parseFloat(draggedNote.dataset.start));

        // Snap to grid (round to nearest 0.25 beats)
        const snappedStartTime = Math.round(startTime);
        const snappedEndTime = Math.round(endTime);

        const width = (snappedEndTime - snappedStartTime);
        //alert(width);

        // Calculate new note value based on vertical drag position
        const noteValue = Math.round(snappedTop / 25); // Assuming 25px = 1 note

        // Ensure the note stays within the container's boundaries
        const maxLeft = dawContainer.clientWidth - draggedNote.offsetWidth;
        const maxTop = dawContainer.clientHeight - draggedNote.offsetHeight;

        // Snap to grid (round to nearest note value)
        const snappedNoteValue = Math.min(Math.max(0, Math.round(noteValue)), parseInt(Math.round(maxTop / 25)));

        newLeft = Math.min(Math.max(snappedLeft, 0), maxLeft);
        newTop = Math.min(Math.max(snappedTop, 0), maxTop);

        // Update note position
        draggedNote.style.left = `${newLeft}px`;
        draggedNote.style.top = `${newTop}px`;
        draggedNote.dataset.start = parseInt(Math.round(newLeft / 25));
        draggedNote.dataset.end = parseInt((newLeft)/25 + width);
        draggedNote.dataset.note = snappedNoteValue;
    }
});

document.addEventListener('mouseup', (e) => {
    if (isDragging || isResizing) {
        // Update the melody array
        updateMelodyArray();
        isDragging = false;
        isResizing = false;
        draggedNote = null;
        resizeHandle = null;
        e.preventDefault();
    }
});

function updateMelodyArray() {
    const notes = dawContainer.getElementsByClassName('note');
    melody = [];
    for (const note of notes) {
        const noteValue = parseInt(note.dataset.note);
        const startTime = parseInt(note.dataset.start);
        const endTime = parseInt(note.dataset.end) + 1;
        melody.push([noteValue, startTime, endTime]);
    }
    console.log('Updated melody:' + melody);
}

// Position notes based on their start and end times
const notes = dawContainer.getElementsByClassName('note');
for (const note of notes) {
    const noteValue = parseInt(note.dataset.note);
    const startTime = parseInt(note.dataset.start);
    const endTime = parseInt(note.dataset.end);
    note.style.left = `${startTime * 25}px`;
    note.style.top = `${noteValue * 25}px`;
    note.style.width = `${(endTime - startTime) * 25 + 4}px`;
    note.draggable = true;
}
