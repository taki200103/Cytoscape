package org.cytoscape.sample.internal;

import org.cytoscape.work.AbstractTask;
import org.cytoscape.work.TaskMonitor;

import javax.swing.*;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

public class AddNodeTask extends AbstractTask {
    private static final String OUTPUT_PATH = "E:\\OOP\\Project_OOP\\Python\\link_process\\link_nodesview.csv";
    private final File inputCsvFile;

    public AddNodeTask(File inputCsvFile) {
        this.inputCsvFile = inputCsvFile;
    }

    @Override
    public void run(TaskMonitor taskMonitor) throws Exception {
        taskMonitor.setTitle("Saving CSV File");

        // Kiểm tra xem file đầu vào có tồn tại và hợp lệ không
        if (inputCsvFile == null || !inputCsvFile.exists() || !inputCsvFile.isFile()) {
            throw new IOException("Input CSV file not found or invalid.");
        }

        // Lưu file đầu vào về đường dẫn chỉ định
        saveCsvFile(taskMonitor);

        // Hiển thị thông báo thành công sau khi lưu file
        showSuccessMessage();
    }

    private void saveCsvFile(TaskMonitor taskMonitor) throws IOException {
        Path outputPath = Path.of(OUTPUT_PATH);

        // Kiểm tra và tạo thư mục nếu chưa tồn tại
        Path parentDir = outputPath.getParent();
        if (!Files.exists(parentDir)) {
            Files.createDirectories(parentDir);
            taskMonitor.setStatusMessage("Created directory: " + parentDir.toString());
        }

        // Copy file CSV từ đường dẫn đầu vào đến đường dẫn đích
        try {
            Files.copy(inputCsvFile.toPath(), outputPath, java.nio.file.StandardCopyOption.REPLACE_EXISTING);
            taskMonitor.setStatusMessage("File saved successfully to: " + OUTPUT_PATH);
        } catch (IOException e) {
            taskMonitor.setStatusMessage("Failed to save file: " + e.getMessage());
            throw e;
        }
    }

    private void showSuccessMessage() {
        // Hiển thị thông báo thành công với JOptionPane
        SwingUtilities.invokeLater(() -> {
            JOptionPane.showMessageDialog(
                null,
                "File saved successfully to:\n" + OUTPUT_PATH,
                "Save CSV File",
                JOptionPane.INFORMATION_MESSAGE
            );
        });
    }
}
