import {
  Accordion,
  AccordionItem,
  AccordionButton,
  AccordionPanel,
  AccordionIcon,
  Box,
  List,
  ListItem,
  Text,
  HStack,
} from '@chakra-ui/react';

const ChapterList = () => {
  const lists = [
    {
      id: 0,
      name: "Artificial Intelligence",
      items: [
        { id: 1, name: "What's AI" },
        { id: 2, name: "Machine Learning" },
        { id: 3, name: "Deep Learning" },
        { id: 4, name: "Neural Network" },
        { id: 5, name: "ML Frameworks and Libraries" },
        { id: 6, name: "PyTorch" },
        { id: 7, name: "Prerequisites" },
        { id: 8, name: "How to take this Course" },
      ],
    },
    { id: 1, name: "TENSORs" },
    { id: 2, name: "A Straight Line Model" },
    { id: 3, name: "A Binary Classification Model" },
    { id: 4, name: "A CNN Model" },
    { id: 5, name: "A Vision Transformer Model" },
    { id: 6, name: "A Language Translation Model" },
    { id: 7, name: "A Speech Recognition Model" },
    { id: 8, name: "A Speech Generation Model" },
    { id: 9, name: "A Recommendation Model" },
  ];

  return (
    <Accordion allowToggle>
      <List spacing={3} py={2} px={4}>
        <ListItem>
          <Text fontSize='md'>Introduction</Text>
        </ListItem>
      </List>
      {lists.map((list) => (
        <AccordionItem key={list.id}>
          <AccordionButton>
            <Box flex='1' textAlign='left'>
              <HStack spacing={3}>
                <Text>{list.id}</Text>
                <Text fontSize='md'>{list.name}</Text>
              </HStack>
            </Box>
            <AccordionIcon />
          </AccordionButton>
          <AccordionPanel pb={4}>
            <List spacing={3}>
              {list.items?.map((item) => (
                <ListItem key={item.id}>
                  <Text fontSize='sm'>{item.name}</Text>
                </ListItem>
              ))}
            </List>
          </AccordionPanel>
        </AccordionItem>
      ))}
        <List spacing={3} px={4} py={2}>
          <ListItem>
            <Text fontSize='md'>Dedication</Text>
          </ListItem>
          <ListItem>
            <Text fontSize='md'>Additional Resources</Text>
          </ListItem>
          <ListItem>
            <Text fontSize='md'>Credits</Text>
          </ListItem>
        </List>
    </Accordion>
  );
};

export default ChapterList;