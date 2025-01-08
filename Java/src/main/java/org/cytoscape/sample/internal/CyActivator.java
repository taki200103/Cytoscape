package org.cytoscape.sample.internal;

import org.cytoscape.service.util.AbstractCyActivator;
import org.cytoscape.application.CyApplicationManager;
import org.cytoscape.view.model.CyNetworkViewManager;
import org.cytoscape.view.vizmap.VisualMappingManager;
import org.cytoscape.view.vizmap.VisualMappingFunctionFactory;
import org.osgi.framework.BundleContext;

import java.util.Properties;
import java.util.logging.Logger;

public class CyActivator extends AbstractCyActivator {
    private static final Logger logger = Logger.getLogger(CyActivator.class.getName());

    @Override
    public void start(BundleContext context) {
        logger.info("Starting CyActivator...");

        // Lấy các dịch vụ cần thiết từ context
        CyApplicationManager applicationManager = getService(context, CyApplicationManager.class);
        CyNetworkViewManager networkViewManager = getService(context, CyNetworkViewManager.class);
        VisualMappingManager visualMappingManager = getService(context, VisualMappingManager.class);
        VisualMappingFunctionFactory discreteMappingFactory = getService(context, VisualMappingFunctionFactory.class);

        // Khởi tạo và đăng ký NodeClassificationTaskFactory
        NodeClassificationTaskFactory classificationTaskFactory = new NodeClassificationTaskFactory(applicationManager, networkViewManager, visualMappingManager, discreteMappingFactory);
        Properties classificationProps = new Properties();
        classificationProps.put("preferredMenu", "Apps.MyApp");
        classificationProps.put("title", "Node Classification");
        registerService(context, classificationTaskFactory, org.cytoscape.work.TaskFactory.class, classificationProps);
        logger.info("Registered NodeClassificationTaskFactory with title 'Node Classification'.");

        // Khởi tạo và đăng ký GroupNodesTaskFactory
        GroupNodesTaskFactory groupTaskFactory = new GroupNodesTaskFactory(applicationManager, networkViewManager);
        Properties groupProps = new Properties();
        groupProps.put("preferredMenu", "Apps.MyApp");
        groupProps.put("title", "Group Nodes by Label");
        registerService(context, groupTaskFactory, org.cytoscape.work.TaskFactory.class, groupProps);
        logger.info("Registered GroupNodesTaskFactory with title 'Group Nodes by Label'.");

        // Khởi tạo và đăng ký NodeEmbeddingTaskFactory
        NodeEmbeddingTaskFactory embeddingTaskFactory = new NodeEmbeddingTaskFactory();
        Properties embeddingProps = new Properties();
        embeddingProps.put("preferredMenu", "Apps.MyApp");
        embeddingProps.put("title", "Node Embedding");
        registerService(context, embeddingTaskFactory, org.cytoscape.work.TaskFactory.class, embeddingProps);
        logger.info("Registered NodeEmbeddingTaskFactory with title 'Node Embedding'.");

        // Khởi tạo và đăng ký AddNodeTaskFactory
        AddNodeTaskFactory addNodeTaskFactory = new AddNodeTaskFactory();
        Properties addNodeProps = new Properties();
        addNodeProps.put("preferredMenu", "Apps.MyApp");
        addNodeProps.put("title", "Add Node");
        registerService(context, addNodeTaskFactory, org.cytoscape.work.TaskFactory.class, addNodeProps);
        logger.info("Registered AddNodeTaskFactory with title 'Add Node'.");

        // Khởi tạo và đăng ký AddViewTaskFactory (Nút Add View)
        AddViewTaskFactory addViewTaskFactory = new AddViewTaskFactory(applicationManager, networkViewManager);
        Properties addViewProps = new Properties();
        addViewProps.put("preferredMenu", "Apps.MyApp");
        addViewProps.put("title", "Add View");
        registerService(context, addViewTaskFactory, org.cytoscape.work.TaskFactory.class, addViewProps);
        logger.info("Registered AddViewTaskFactory with title 'Add View'.");

        // Khởi tạo và đăng ký LinkPredictionTaskFactory (Nút Link Prediction)
        LinkPredictionTaskFactory linkPredictionTaskFactory = new LinkPredictionTaskFactory();
        Properties linkPredictionProps = new Properties();
        linkPredictionProps.put("preferredMenu", "Apps.MyApp");
        linkPredictionProps.put("title", "Link Prediction");
        registerService(context, linkPredictionTaskFactory, org.cytoscape.work.TaskFactory.class, linkPredictionProps);
        logger.info("Registered LinkPredictionTaskFactory with title 'Link Prediction'.");
    }
}
