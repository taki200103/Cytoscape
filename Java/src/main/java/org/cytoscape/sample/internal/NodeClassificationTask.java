package org.cytoscape.sample.internal;

import org.cytoscape.work.AbstractTask;
import org.cytoscape.work.TaskMonitor;
import org.cytoscape.application.CyApplicationManager;
import org.cytoscape.model.CyNetwork;
import org.cytoscape.view.model.CyNetworkViewManager;
import org.cytoscape.view.model.CyNetworkView;
import org.cytoscape.view.presentation.property.BasicVisualLexicon;
import org.cytoscape.view.vizmap.VisualMappingManager;
import org.cytoscape.view.vizmap.VisualStyle;
import org.cytoscape.view.vizmap.VisualMappingFunctionFactory;
import org.cytoscape.view.vizmap.mappings.DiscreteMapping;
import java.awt.Paint;
import java.awt.Color;
import java.util.HashMap;
import java.util.Map;

public class NodeClassificationTask extends AbstractTask {
    private final CyApplicationManager applicationManager;
    private final CyNetworkViewManager networkViewManager;
    private final VisualMappingManager visualMappingManager;
    private final VisualMappingFunctionFactory discreteMappingFactory;

    public NodeClassificationTask(CyApplicationManager applicationManager, CyNetworkViewManager networkViewManager, VisualMappingManager visualMappingManager, VisualMappingFunctionFactory discreteMappingFactory) {
        this.applicationManager = applicationManager;
        this.networkViewManager = networkViewManager;
        this.visualMappingManager = visualMappingManager;
        this.discreteMappingFactory = discreteMappingFactory;
    }

    @Override
    public void run(TaskMonitor taskMonitor) {
        taskMonitor.setTitle("Node Classification Task");
        taskMonitor.setProgress(0.0);
        taskMonitor.setStatusMessage("Đang kiểm tra mạng...");

        CyNetwork network = applicationManager.getCurrentNetwork();
        if (network == null) {
            return;
        }

        taskMonitor.setProgress(0.2);
        taskMonitor.setStatusMessage("Đang lấy network view...");

        CyNetworkView networkView = networkViewManager.getNetworkViews(network).iterator().next();
        if (networkView == null) {
            return;
        }

        taskMonitor.setProgress(0.4);
        taskMonitor.setStatusMessage("Đang tạo bảng màu...");

        // Create a color map for labels 0 to 39
        Map<Integer, Paint> colorMap = new HashMap<>();
        for (int i = 0; i < 40; i++) {
            colorMap.put(i, new Color((i * 32) % 256, (i * 64) % 256, (i * 128) % 256));
        }

        taskMonitor.setProgress(0.6);
        taskMonitor.setStatusMessage("Đang áp dụng màu sắc...");

        DiscreteMapping<Integer, Paint> mapping = (DiscreteMapping<Integer, Paint>) discreteMappingFactory.createVisualMappingFunction("label", Integer.class, BasicVisualLexicon.NODE_FILL_COLOR);
        for (Map.Entry<Integer, Paint> entry : colorMap.entrySet()) {
            mapping.putMapValue(entry.getKey(), entry.getValue());
        }

        taskMonitor.setProgress(0.8);
        taskMonitor.setStatusMessage("Đang cập nhật giao diện...");

        VisualStyle style = visualMappingManager.getCurrentVisualStyle();
        style.addVisualMappingFunction(mapping);
        style.apply(networkView);
        networkView.updateView();

        taskMonitor.setProgress(1.0);
        taskMonitor.setStatusMessage("Hoàn thành!");
    }
}